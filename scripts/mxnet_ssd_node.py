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
        # convert these to subscribed topics

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/img')
        self.timer = self.load_param('~throttle_timer', '5')
        self.threshold = self.load_param('~threshold', 0.5)
        self.start_enabled = self.load_param('~start_enabled ', 'false')
        self.start_zoom_enabled = self.load_param('~start_zoom_enabled ', 'false')

        # crop pattern
        self.level0_ncrops = self.load_param('~level0_ncrops',1)
        self.level1_xcrops = self.load_param('~level1_xcrops',0)
        self.level1_ycrops = self.load_param('~level1_ycrops',0)
        self.level1_crop_size = self.load_param('~level1_crop_size',300)
        
        # location of mxnet model and name, epoch, GPU and number of classes
        self.model_name = self.load_param('~model_name','mobilenet-ssd')
        self.model_directory = self.load_param('~model_directory', '~/mxnet_ssd/')
        self.model_epoch = self.load_param('~model_epoch', '1')
        self.enable_gpu = self.load_param('~enable_gpu', 'false')
        self.num_classes = self.load_param('~num_classes',20)
        self.batch_size = self.load_param('~batch_size',1)

        class_names = 'goose, person, golfcart, lawnmower, dog'
        class_names = 'aeroplane, bicycle, bird, boat, bottle, bus, \
                       car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                       person, pottedplant, sheep, sofa, train, tvmonitor'

        # COMING SOON SECTION
        # save detections output and location
        #self.save_detections = self.load_param('~save_detections', 'false')
        #self.save_directory = self.load_param('~save_directory', '/tmp')
        #self.mask_topic = self.load_param('~mask_topic', '/img_segmentations')

        # Class Variables
        self.detection_seq = 0
        self.camera_frame = "camera_frame"
        print self.model_name
        print self.model_directory
        self.classifier = MxNetSsdClassifier(self.model_name, self.model_epoch, self.model_directory, self.batch_size, self.enable_gpu, self.num_classes)
        self.last_detection_time = 0     
    
        # ROS Subscribers
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_enable = rospy.Subscriber('~enable', Bool, self.enable_cb, queue_size=1)

        # ROS Publishers
        detection_topic = self.image_topic +rospy.get_name() + '/detections'
        self.pub_detections=rospy.Publisher(detection_topic, Detection2DArray, queue_size=10)


    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MxNet] %s: %s", param, new_param)
        return new_param


    def enable_cb(self, msg):
        self.start_enabled = msg.data


    def image_cb(self, image):
        if self.start_enabled:
            current_time = rospy.get_rostime().secs
            if current_time % self.timer == 0 and self.last_detection_time != current_time:
                self.last_detection_time = current_time
                try:
                    bridge = CvBridge()
                    cv2_img = bridge.imgmsg_to_cv2(image, "rgb8")
                    frame=np.asarray(cv2_img).copy()
                    data_shape=frame.shape
                    ydim,xdim=frame.shape
                    min_dim=np.min(data_shape[0:2])
                    max_dim=np.max(data_shape[0:2])

                    framelist=[]
                    # decision tree for crop-pattern. First-level crop pattern:
                    if (self.level0_ncrops==0):
                        # resize to square
                        framelist.append(cv2.resize(frame,(min_dim,min_dim)))
                    elif (self.level0_ncrops==1):
                        # center crop to square along the larger dimension (note, if dims are equal, do nothing)
                        if (ydim < xdim):
                            framelist.append(frame[:,(max_dim-min_dim)/2:-(max_dim-min_dim)/2,:])
                        elif (xdim < ydim):
                            framelist.append(frame[(max_dim-min_dim)/2:-(max_dim-min_dim)/2,:,:])
                    elif (self.level0_ncrops>1):
                        # crop along larger dimension with ncrops - use the half-crop width (minimum dimension/2) 
                        half_crop=int(round(min_dim/2.0))
                        # set centroids on linear spaced pattern from half_crop to end-half_crop
                        centroids=map(int,np.linspace(half_crop,max_dim-half_crop, self.level0_ncrops))
                        for i in range(0,self.level0_ncrops):
                            if (ydim < xdim):
                                framelist.append(np.copy(frame[:,centroids[i]-half_crop:centroids[i]+half_crop,:]))
                            elif (ydim < xdim):
                                framelist.append(np.copy(frame[centroids[i]-half_crop:centroids[i]+half_crop,:,:]))

                    # Second-level crop pattern:
                    if (self.level1_xcrops>0 or self.level1_ycrops>0):
                        # get half crop sizes
                        half_crop=int(round(self.level1_crop_size/2.0))
                        # set centroids on linear spaced pattern from half_crop to end-half_crop
                        xcentroids=map(int,np.linspace(half_crop,xdim-half_crop, self.level1_xcrops))
                        ycentroids=map(int,np.linspace(half_crop,ydim-half_crop, self.level1_ycrops))
                        for i in range(0,self.level1_xcrops):
                            for j in range(0,self.level1_ycrops):
                                framelist.append(np.copy(frame[ycentroids[i]-half_crop:ycentroids[i]+half_crop,xcentroids[i]-half_crop:xcentroids[i]+half_crop,:]))

                    # pass all frame crops to classifier
                    all_detections=self.classifier.detect(framelist, self.threshold)

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

