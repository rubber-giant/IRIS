#!/usr/bin/env python3

import time
import cv2
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from msg import integer
from ultralytics import YOLO
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.motor_control_pub = self.create_publisher(int, 'integer', QoSProfile(depth=10))

        self.model = YOLO("yolov8s.pt")
        self.cam = cv2.VideoCapture(0)

    def detect_objects(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if ret :
                cropped_frame = frame[0:479, 140:500]
                result = self.model.predict(cropped_frame, conf=0.75)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])

                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    area = width * height
                    self.stage = "close" if area > 105000 else "far"

                    if self.stage == "close":
                        name = str(int(box.cls))
                        cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(cropped_frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.imshow("test", frame)

            if cv2.waitKey(3) & 0xFF == ord('q'): 
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Perform object detection
        self.detect_objects()

        # Process detected objects and send motor control commands
        self.process_detected_objects()

    def set_vibration_intensity(self ,intensity):
        self.motor_control_pub.publish(intensity)


    def process_detected_objects(self):
        # Process detected objects and determine motor control commands
        if self.stage == "far":
            self.set_vibration_intensity(10)
            time.sleep(1.5)
            self.set_vibration_intensity(0)
        elif self.stage == "close":
            self.set_vibration_intensity(80)
            time.sleep(2.25)
            self.set_vibration_intensity(0)
        else:
            self.set_vibration_intensity(0)

        # For demonstration, we'll simply vibrate the motors if a person is detected
        self.set_vibration_intensity(50)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # Clean up
    node.cam.release()
    cv2.destroyAllWindows()

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
