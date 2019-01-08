#!/usr/bin/env python

import rospy
import os
import time
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge, CvBridgeError
from mxnet_depseg_hydra import MxNetDepthSegmentationHydra
from mxnet_segmentation_custom_functions import SegmentationFunctions

class RosMxNetDepthSegmentation:

    def __init__(self):
        rospy.logwarn("Initializing")        

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/usb_cam/image_raw')
        self.timer = self.load_param('~throttle_timer', 0)
        self.latency_threshold_time=self.load_param('~latency_threshold', 0.1)
        self.run_continuous = self.load_param('~run_continuous ', True)
        self.avg_segmentation_frames = self.load_param('~avg_segmentation_frames', 1)
        self.overlay_topic = self.load_param('~overlay_topic', '~segmentation_overlay')
        self.mask_topic = self.load_param('~mask_topic', '~segmentation_mask')
        self.depth_topic = self.load_param('~depth_topic', '~depth')
        self.mask_values=str(self.load_param('~mask_values', '12'))
        # resize should ideally be in multiples of 32, from 2-10 they are: 64, 96, 128, 160, 192, 224, 256, 288, 320
        #self.image_resize = map(int,self.load_param('~image_resize', '256x512').split('x'))
        self.image_resize = map(int,self.load_param('~image_resize', '512x256').split('x'))

        # mxnet model name, GPU
        self.enable_gpu = self.load_param('~enable_gpu', True)
        self.model_filename = self.load_param('~model_filename','UResNetHydra18_57_0050.params')
        self.model_dir = self.load_param('~model_dir', os.environ['HOME']+'/mxnet_ssd/')

        # Class Variables
        self.run_once = False
        self.frame_counter=0
        self.detection_seq = 0
        self.camera_frame = "camera_frame"
        self.mask_values = [int(c.strip(' ')) for c in self.mask_values.strip('\n').split(',')]
        self.segmentation_utils = SegmentationFunctions(mask_values=self.mask_values)
        self.last_detection_time = 0     
        self.reported_overlaps=False
        self.data_shape=None
        self.image_counter=0
        self.hydra=None
        print self.image_resize[::-1]
        self.hydra = MxNetDepthSegmentationHydra(self.model_dir, self.model_filename, resnet_depth=18, seg_classes=10, gpu_enabled=self.enable_gpu, image_resize=self.image_resize[::-1])

        # ROS Subscribers
        self.start_time=time.time()
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_run_once = rospy.Subscriber('~run_once',Bool, self.run_once_cb, queue_size=1)
        self.sub_run_continuous = rospy.Subscriber('~run_continuous',Bool, self.run_continuous_cb, queue_size=1)

        # ROS Publishers
        self.pub_overlay=rospy.Publisher(self.overlay_topic, Image, queue_size=1)
        self.pub_mask=rospy.Publisher(self.mask_topic, Image, queue_size=1, latch=True)
        self.pub_depth=rospy.Publisher(self.depth_topic, Image, queue_size=1)

        rospy.loginfo("[MxNet Initialized with model %s]",self.model_filename)
        
    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MxNet] %s: %s", param, new_param)
        return new_param

    def run_once_cb(self, msg):
        if (type(msg)==type(True)):
            self.run_once = msg
        else:
            self.run_once = msg.data
        rospy.loginfo("MxNet run_once_cb: "+str(self.run_once))

    def run_continuous_cb(self, msg):
        if (type(msg)==type(True)):
            temp_run_continuous = msg
        else:
            temp_run_continuous = msg.data
        self.run_continuous = temp_run_continuous
        rospy.loginfo("MxNet run_continuous_cb: "+str(self.run_continuous))

    def image_cb(self, image):
        # check latency and return if more than latency threshold out of date
        current_time = rospy.get_rostime().secs + float(int(rospy.get_rostime().nsecs/1000000.0))/1000.0
        image_time = image.header.stamp.secs + float(int(image.header.stamp.nsecs/1000000.0))/1000.0
        if (current_time-image_time>self.latency_threshold_time):
            return

        if self.run_continuous or self.run_once:
            current_time = rospy.get_rostime().secs
            #print current_time
            if self.last_detection_time+self.timer <= current_time:
                self.last_detection_time = current_time
                try:
                    cv2_img = self.bridge.imgmsg_to_cv2(image, "rgb8")
                    frame=np.asarray(cv2_img).copy()
                    self.image_counter=self.image_counter+1
                    if (self.image_counter % 11) == 10:
                        rospy.loginfo("Images processed per second=%.2f", float(self.image_counter)/(time.time() - self.start_time))

                    # produce segmentation (seg) and overlay on the frame
                    seg, depth = self.hydra.segment_and_depth(frame)

                    # create the segmentation mask and overlay on the frame
                    mask = self.segmentation_utils.segmentation_to_mask(seg)
                    overlay = self.segmentation_utils.overlay_mask(frame, mask, mask_color=[255,0,0],alpha=0.5)
                    mask = (255.0*seg/np.max(seg)).astype(np.uint8)

                    self.run_once = False

                    # invert depth about 1 to turn into approx real depths
                    #depth[depth>0] = 1./depth[depth>0]
                    # scale up to 255
                    #depth=(255.0*depth/np.max(depth)).astype(np.uint8)

                    try:
                        # send uncompressed image
                        self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, "rgb8"))
                        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
                        self.pub_depth.publish(self.bridge.cv2_to_imgmsg(depth, "mono8"))
                    except CvBridgeError as e:
                        print(e)

                except CvBridgeError, e:
                    rospy.logerr(e)


if __name__ == '__main__':
    rospy.init_node("rr_mxnet_depseg_hydra", anonymous=False, log_level=rospy.INFO)
    ros_mxnet_depseg_hydra = RosMxNetDepthSegmentation()
    rospy.spin()

