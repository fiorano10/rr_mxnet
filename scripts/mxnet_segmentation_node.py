#!/usr/bin/env python

import rospy
import time
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge, CvBridgeError
from mxnet_segmentation import MxNetSegmentation
from mxnet_segmentation_custom_functions import SegmentationFunctions

class RosMxNetSegmentation:

    def __init__(self):
        rospy.logwarn("Initializing")        

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/usb_cam/image_raw')
        self.timer = self.load_param('~throttle_timer', 2)
        self.latency_threshold_time=self.load_param('~latency_threshold', 1)
        self.run_continuous = self.load_param('~run_continuous ', False)
        self.avg_segmentation_frames = self.load_param('~avg_segmentation_frames', 1)
        self.overlay_topic = self.load_param('~overlay_topic', '~segmentation_overlay')
        self.mask_topic = self.load_param('~mask_topic', '~segmentation_mask')
        self.mask_values=str(self.load_param('~mask_values', '12'))

        # mxnet model name, GPU
        self.enable_gpu = self.load_param('~enable_gpu', False)
        self.network = self.load_param('~network','deeplab_resnet50_ade')

        # Class Variables
        self.run_once = False
        self.frame_counter=0
        self.detection_seq = 0
        self.camera_frame = "camera_frame"
        self.mask_values = [int(c.strip(' ')) for c in self.mask_values.strip('\n').split(',')]
        self.segmenter = MxNetSegmentation(None, None, self.network, self.enable_gpu, image_resize=300)
        self.segmentation_utils = SegmentationFunctions(mask_values=self.mask_values)
        self.last_detection_time = 0     
        self.reported_overlaps=False
        self.data_shape=None
        self.image_counter=0

        # ROS Subscribers
        self.start_time=time.time()
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_run_once = rospy.Subscriber('~run_once',Bool, self.run_once_cb, queue_size=1)
        self.sub_run_continuous = rospy.Subscriber('~run_continuous',Bool, self.run_continuous_cb, queue_size=1)

        # ROS Publishers
        self.pub_overlay=rospy.Publisher(self.overlay_topic, Image, queue_size=1)
        self.pub_mask=rospy.Publisher(self.mask_topic, Image, queue_size=1, latch=True)

        rospy.loginfo("[MxNet Initialized with model %s]",self.network)
        
    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MxNet] %s: %s", param, new_param)
        return new_param

    def run_once_cb(self, msg):
        self.run_once = msg.data
        rospy.loginfo("MxNet run_once_cb: "+str(self.run_once))

    def run_continuous_cb(self, msg):
        self.run_continuous = msg.data
        rospy.loginfo("MxNet run_continuous_cb: "+str(self.run_continuous))

    def image_cb(self, image):
        # check latency and return if more than latency threshold out of date
        current_time = rospy.get_rostime().secs + float(int(rospy.get_rostime().nsecs/1000000.0))/1000.0
        image_time = image.header.stamp.secs + float(int(image.header.stamp.nsecs/1000000.0))/1000.0
        if (current_time-image_time>self.latency_threshold_time):
            return

        if self.run_continuous or self.run_once:
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
                    seg = self.segmenter.segment(frame)

                    # create the segmentation mask and overlay on the frame
                    mask = self.segmentation_utils.segmentation_to_mask(seg)
                    overlay = self.segmentation_utils.overlay_mask(frame, mask, mask_color=[255,0,0],alpha=0.5)

                    self.run_once = False

                    # if specified, publish segmented images
                    try:
                        # send uncompressed image
                        self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, "rgb8"))
                        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
                    except CvBridgeError as e:
                        print(e)

                except CvBridgeError, e:
                    rospy.logerr(e)


if __name__ == '__main__':
    rospy.init_node("rr_mxnet_segmentation", anonymous=False, log_level=rospy.INFO)
    ros_mxnet_segmentation = RosMxNetSegmentation()
    rospy.spin()

