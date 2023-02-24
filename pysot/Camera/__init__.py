import time
import numpy as np
import glob
import os
import cv2

RGB=0
RGBD=1
ROS=2
FILE=3
FILE_ADDRESS = '/imgs'
FILE_TYPE = 'jpg'

class Camera(object):
    def __init__(self, sourse_type, process_function=None):
        self.sourse_type = sourse_type
        self.camera = None
        self.process = process_function
        self.start()
        

    def start(self):
        if self.sourse_type == RGB:
            self.camera = cv2.VideoCapture(0)
        elif self.sourse_type == RGBD:
            import pyrealsense2 as rs
            self.camera = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.camera.start(config)
        elif self.sourse_type == ROS:
            import rospy
            from cv_bridge import CvBridge, CvBridgeError
            from sensor_msgs.msg import Image, CompressedImage
            
            self.topic_name = '/camera/color/image_raw/compressed'
            self.bridge = CvBridge()
        elif self.sourse_type == FILE:
            print(FILE_ADDRESS+'/*.'+FILE_TYPE)
            self.camera = sorted(glob.glob(FILE_ADDRESS+'/*.'+FILE_TYPE))
            print('len', len(self.camera))

    def get_img(self,data=None):
        if self.sourse_type == RGB:
            if self.camera.isOpened():
                ret, frame = self.camera

        elif self.sourse_type == RGBD:
            # print('wait for frames')
            frames = self.camera.wait_for_frames()
            
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            
        elif self.sourse_type == ROS:
            frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        elif self.sourse_type == FILE:
            # print(len(self.camera))
            if len(self.camera):
                frame = cv2.imread(self.camera[0])
                # print(self.camera[0])
                del self.camera[0]
            else:
                
                frame = None
        return frame

    def callback(self, data):
        img = self.get_img(data)
        if img is None:
            if not self.sourse_type == ROS:
                return False
            else:
                os.system('kill ' + str(os.getpid()))
        if not self.process == None:
            if not self.process(img):
                
                if self.sourse_type == ROS:
                    os.system('kill ' + str(os.getpid()))
                else:
                    return False
            # print('yes')
        if not self.sourse_type == ROS:
            return True

    def run(self):
        if self.sourse_type == ROS:
            rospy.init_node('webcam_display', anonymous=True)
            rospy.Subscriber(self.topic_name, CompressedImage, self.callback)
            rospy.spin()
            # pass
        else:
            while self.callback(None):
                pass


if __name__ == '__main__':
    pass
