#!/usr/bin/env python

import rospy
import time
import os
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
from mxnet_seg import MxNetSegmentation

class RosMxNetSeg:

    def __init__(self):
        rospy.logwarn("Initializing")        
        # convert these to subscribed topics

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/usb_cam/image_raw')
        self.timer = self.load_param('~throttle_timer', 5)
        self.latency_threshold_time=self.load_param('~latency_threshold', 2)
        self.start_enabled = self.load_param('~start_enabled ', False)

        # mxnet model name, GPU
        self.enable_gpu = self.load_param('~enable_gpu', True)
        self.batch_size = self.load_param('~batch_size',1)
        self.network = self.load_param('~network','deeplab_resnet50_ade')

        # Digilabs section
        self.batch_size = 1
        self.enable_gpu = False
        self.start_enabled = True
        self.timer = 1
        self.latency_threshold_time=1
        self.start_time=time.time()

        self.mask_topic = self.load_param('~mask_topic', '~segmentation')

        # Class Variables
        self.detection_seq = 0
        self.camera_frame = "camera_frame"
        self.segmenter = MxNetSegmentation(None, None, self.network, self.batch_size, self.enable_gpu, image_resize=300)
        self.last_detection_time = 0     
        self.reported_overlaps=False
        self.data_shape=None
        self.image_counter=0

        self.bridge = CvBridge()
        # ROS Subscribers
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_enable = rospy.Subscriber('~enable', Bool, self.enable_cb, queue_size=1)

        # ROS Publishers
        self.pub_img_detections=rospy.Publisher(self.mask_topic, Image, queue_size=1)

        rospy.loginfo("[MxNet Initialized with model %s]",self.network)
        
    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MxNet] %s: %s", param, new_param)
        return new_param

    def enable_cb(self, msg):
        self.start_enabled = msg.data
        rospy.loginfo("MxNet enable_cb: "+str(self.start_enabled))

    def image_cb(self, image):
        # check latency and return if more than latency threshold out of date
        current_time = rospy.get_rostime().secs + float(int(rospy.get_rostime().nsecs/1000000.0))/1000.0
        image_time = image.header.stamp.secs + float(int(image.header.stamp.nsecs/1000000.0))/1000.0
        if (current_time-image_time>self.latency_threshold_time):
            return

        if self.start_enabled:
            current_time = rospy.get_rostime().secs
            if self.last_detection_time+self.timer <= current_time:
                self.last_detection_time = current_time
                try:
                    cv2_img = self.bridge.imgmsg_to_cv2(image, "rgb8")
                    frame=np.asarray(cv2_img).copy()
                    self.image_counter=self.image_counter+1
                    if (self.image_counter % 11) == 10:
                        rospy.loginfo("Images segmented per second=%.2f", float(self.image_counter)/(time.time() - self.start_time))

                    # produce segmentation (seg) and overlay on the frame
                    frame,seg= self.segmenter.segment(frame)

                    # if specified, publish segmented images
                    try:
                        # send uncompressed image
                        self.pub_img_detections.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
                    except CvBridgeError as e:
                        print(e)

                except CvBridgeError, e:
                    rospy.logerr(e)


if __name__ == '__main__':
    rospy.init_node("mxnet_seg_node", anonymous=False, log_level=rospy.INFO)
    ros_mxnet_seg = RosMxNetSeg()
    rospy.spin()

