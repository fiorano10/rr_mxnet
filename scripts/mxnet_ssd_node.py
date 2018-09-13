#!/usr/bin/env python

import rospy
import sys
import os
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
from mxnet_ssd import MxNetSsdClassifier

class RosMxNetSSD:

    def __init__(self):
        rospy.logwarn("Initializing")        

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/img')
        self.timer = self.load_param('~timer', '5')
        self.save_detections = self.load_param('~save_detections', 'false')
        self.save_directory = self.load_param('~save_directory', '/tmp')
        self.model_name = self.load_param('~model_name')
        self.threshold = self.load_param('~threshold', 0.5)
        self.model_directory = self.load_param('~model_directory', '~/mxnet_ssd/')
        self.enable_gpu = self.load_param('~enable_gpu', 'false')

        # Class Variables
        self.detection_seq = 0
        self.is_enabled = False
        self.camera_frame = "camera_frame"
        print self.model_name
        print self.model_directory
        self.classifier = MxNetSsdClassifier(self.model_name, self.model_directory, self.enable_gpu)
        self.last_detection_time = 0     
    
        # ROS Subscribers
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_enable = rospy.Subscriber('~enable', Bool, self.enable_cb, queue_size=1)

        # ROS Publishers
        detection_topic = self.image_topic +rospy.get_name() + '/detections'
        self.pub_detections=rospy.Publisher(detection_topic, Detection2DArray, queue_size=10)


    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MXNET] %s: %s", param, self.image_topic)
        return new_param


    def enable_cb(self, msg):
        self.is_enabled = msg.data


    def image_cb(self, image):
        if self.is_enabled:
            current_time = rospy.get_rostime().secs
            if current_time % self.timer == 0 and self.last_detection_time != current_time:
                self.last_detection_time = current_time
                try:
                    bridge = CvBridge()
                    cv2_img = bridge.imgmsg_to_cv2(image, "rgb8")
                    frame=np.asarray(cv2_img).copy()
                    frame,all_detections=self.classifier.detect(frame, self.threshold)

                    detections_msg = Detection2DArray()
                    if (len(all_detections)>0):
                        i=0
                        rospy.logwarn("Object detected")
                        for det in all_detections:
                            detection = Detection2D()
                            detection.header.seq = self.detection_seq
                            detection.header.stamp = rospy.Time.now()
                            detection.header.frame_id = self.camera_frame
                            result = [ObjectHypothesisWithPose()]
                            result[0].id = det[0]
                            result[0].score = det[1]
                            detection.results = result
                            detection.bbox.center.x = (det[2]+det[4])/2
                            detection.bbox.center.y = (det[3]+det[5])/2 
                            detection.bbox.size_x = det[4]-det[2]
                            detection.bbox.size_y = det[5]-det[3]
                            detections_msg.detections.append(detection)
                            i+=1
                    self.pub_detections.publish(detections_msg)
                    self.detection_seq += 1
            
                except CvBridgeError, e:
                    rospy.logerr(e)


if __name__ == '__main__':
    rospy.init_node("mxnet_ssd_node", anonymous=False, log_level=rospy.INFO)
    ros_mxnet_ssd = RosMxNetSSD()
    rospy.spin()
    
