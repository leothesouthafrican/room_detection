import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import numpy as np
import cv2
import time
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_save_time = 0  # Initialize last save time
        self.start_time = time.time()  # Record the start time
        self.save_interval = 10  # Save interval in seconds
        self.confidence_threshold = 0.5  # Lowered confidence threshold for testing
        self.save_path = 'saved_people'  # Path to save images
        os.makedirs(self.save_path, exist_ok=True)  # Ensure the save path exists
        # Do not set self.use_frame here

def app_callback(pad, info, user_data):
    print("app_callback called", flush=True)
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        print("Buffer is None", flush=True)
        return Gst.PadProbeReturn.OK

    # Increment frame count
    user_data.increment()
    frame_count = user_data.get_count()

    # Get the caps from the pad
    video_format, width, height = get_caps_from_pad(pad)
    print(f"Format: {video_format} ({type(video_format)}), Width: {width} ({type(width)}), Height: {height} ({type(height)})", flush=True)

    # Get video frame if available
    frame = None
    if user_data.use_frame and video_format and width and height:
        frame = get_numpy_from_buffer(buffer, video_format, width, height)
        if frame is not None:
            print(f"Frame obtained: shape={frame.shape}", flush=True)
        else:
            print("Frame is None after get_numpy_from_buffer", flush=True)
    else:
        print("Frame not obtained due to missing video_format, width, or height", flush=True)
        print(f"user_data.use_frame={user_data.use_frame}, video_format={video_format}, width={width}, height={height}", flush=True)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    print(f"Number of detections: {len(detections)}", flush=True)

    # Process detections
    person_detected = False
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        print(f"Detection: label={label}, confidence={confidence}", flush=True)
        if label == "person" and confidence >= user_data.confidence_threshold:
            print("Person detected!", flush=True)
            person_detected = True
            detection_count += 1
            # Draw bounding box on frame
            if frame is not None:
                # Access bbox coordinates correctly
                x1 = bbox.xmin()
                y1 = bbox.ymin()
                x2 = bbox.xmax()
                y2 = bbox.ymax()
                # Scale coordinates to frame size
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Implement saving logic
    current_time = time.time()
    if person_detected and (current_time - user_data.last_save_time >= user_data.save_interval):
        print("Saving logic triggered", flush=True)
        if frame is not None:
            # Save the frame
            filename = time.strftime("person_%Y%m%d_%H%M%S.jpg", time.localtime())
            filepath = os.path.join(user_data.save_path, filename)
            # Convert frame to BGR before saving
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            user_data.last_save_time = current_time
            # Calculate and print the total running time
            elapsed_time = current_time - user_data.start_time
            print(f"Image saved: {filepath}", flush=True)
            print(f"Total running time: {elapsed_time:.2f} seconds", flush=True)
        else:
            print("Frame is None, cannot save image", flush=True)

    # Optional: Display detection count on frame
    if frame is not None:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        user_data.set_frame(frame)
    else:
        print("Frame is None, cannot display detection count", flush=True)

    return Gst.PadProbeReturn.OK

class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        # Set Hailo parameters based on the model used
        self.batch_size = 1
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(self.current_path, 'resources/libyolo_hailortpp_post.so')

        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        # Set the HEF file path based on the network
        if args.hef_path is not None:
            self.hef_path = args.hef_path
        elif args.network == "yolov8s":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
        else:
            raise ValueError("Invalid network type or HEF path not provided")

        # User-defined label JSON file
        if args.labels_json is not None:
            self.labels_config = f' config-path={args.labels_json} '
        else:
            self.labels_config = ''

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        self.create_pipeline()

        # Connect the app_callback to the identity element
        identity_element = self.pipeline.get_by_name('identity_callback')
        if identity_element:
            identity_element.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, self.app_callback, user_data)
        else:
            print("Error: identity_callback element not found in the pipeline", flush=True)

    def get_pipeline_string(self):
        # Source element for the Raspberry Pi camera
        source_element = (
            "libcamerasrc ! "
            f"video/x-raw, format={self.network_format}, width=1920, height=1080, framerate=30/1 ! "
            + QUEUE("queue_src_scale")
            + "videoscale ! "
            f"video/x-raw, width={self.network_width}, height={self.network_height} ! "
            + QUEUE("queue_src_convert")
            + "videoconvert ! "
            f"video/x-raw, format={self.network_format} ! "
        )
        pipeline_string = (
            source_element
            + QUEUE("queue_hailonet")
            + f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} {self.thresholds_str} ! "
            + QUEUE("queue_hailofilter")
            + f"hailofilter so-path={self.default_postprocess_so} {self.labels_config} ! "
            + QUEUE("queue_user_callback")
            + "identity name=identity_callback ! "
            + QUEUE("queue_hailooverlay")
            + "hailooverlay ! "
            + QUEUE("queue_videoconvert")
            + "videoconvert ! "
            + "videoflip method=horizontal-flip ! "
            + QUEUE("queue_hailo_display")
            + "autovideosink sync=false"
        )

        print("GStreamer Pipeline:\n", pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)

    # Create an instance of the user app callback class
    user_data = user_app_callback_class()

    # Parse command-line arguments
    parser = get_default_parser()
    parser.add_argument(
        "--network",
        default="yolov8s",
        choices=['yolov8s'],
        help="Which network to use",
    )
    parser.add_argument(
        "--hef-path",
        default=None,
        help="Path to HEF file",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Path to custom labels JSON file",
    )
    args = parser.parse_args()

    # Create and run the application
    app = GStreamerDetectionApp(args, user_data)

    # Set use_frame to True after app initialization
    user_data.use_frame = True

    app.run()
