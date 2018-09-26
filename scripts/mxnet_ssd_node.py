#!/usr/bin/env python

import rospy
import os
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
from mxnet_ssd import MxNetSSDClassifier
from mxnet_ssd_custom_functions import SSDCropPattern, convert_frame_to_jpeg_string

class RosMxNetSSD:

    def __init__(self):
        rospy.logwarn("Initializing")        
        # convert these to subscribed topics

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/usb_cam/image_raw')
        self.detections_topic = self.load_param('~detections_topic', '~detections')
        self.publish_detection_images = self.load_param('~publish_detection_images', False)
        self.image_detections_topic = self.load_param('~image_detections_topic', '~image')
        self.timer = self.load_param('~throttle_timer', 5)
        self.latency_threshold_time=self.load_param('~latency_threshold', 2)
        self.threshold = self.load_param('~threshold', 0.5)
        self.start_enabled = self.load_param('~start_enabled ', False)
        self.zoom_enabled = self.load_param('~start_zoom_enabled ', False)

        # crop pattern
        self.level0_ncrops = self.load_param('~level0_ncrops',2)
        self.level1_xcrops = self.load_param('~level1_xcrops',4)
        self.level1_ycrops = self.load_param('~level1_ycrops',2)
        self.level1_crop_size = self.load_param('~level1_crop_size',380)
        
        # location of mxnet model and name, epoch, GPU and number of classes
        self.model_directory = self.load_param('~model_directory', os.environ['HOME']+'/mxnet_ssd/')
        self.num_classes = self.load_param('~num_classes',20)
        self.enable_gpu = self.load_param('~enable_gpu', True)
        # recommendation to use self.level0_ncrops in most cases for batch_size
        self.batch_size = self.load_param('~batch_size',1)
        # this bool is only set to True for quicker debugging, in most cases you should use rosparams to set these values explicitly
        use_vocdev_mobilenet=True
        if (use_vocdev_mobilenet):
            self.model_name = self.load_param('~model_name','mobilenet-ssd-512')
            self.model_epoch = self.load_param('~model_epoch', 1)
            self.network = self.load_param('~network','mobilenet')
        else:
            self.model_name = self.load_param('~model_name','ssd_resnet50_512')
            self.model_epoch = self.load_param('~model_epoch', 222)
            self.network = self.load_param('~network','resnet50')
        class_names = 'aeroplane, bicycle, bird, boat, bottle, bus, \
                       car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                       person, pottedplant, sheep, sofa, train, tvmonitor'

        # save detections output and location
        self.save_detections = self.load_param('~save_detections', False)
        self.save_directory = self.load_param('~save_directory', '/tmp')

        # COMING SOON SECTION
        #self.mask_topic = self.load_param('~mask_topic', '/img_segmentations')

        # Class Variables
        self.detection_seq = 0
        self.camera_frame = "camera_frame"
        self.classifier = MxNetSSDClassifier(self.model_name, self.model_epoch, self.model_directory, self.network, self.batch_size, self.enable_gpu, self.num_classes)
        self.imageprocessor = SSDCropPattern(self.zoom_enabled, self.level0_ncrops, self.level1_xcrops, self.level1_ycrops, self.level1_crop_size)
        self.last_detection_time = 0     
        self.reported_overlaps=False
        self.data_shape=None
        self.class_names = [cls for cls in class_names.strip('\n').replace(' ','').split(',')]

        self.bridge = CvBridge()
        # ROS Subscribers
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_enable = rospy.Subscriber('~enable', Bool, self.enable_cb, queue_size=1)
        self.sub_zoom = rospy.Subscriber('~zoom', Bool, self.zoom_cb, queue_size=1)

        # ROS Publishers
        self.pub_detections=rospy.Publisher(self.detections_topic, Detection2DArray, queue_size=10)
        if (self.publish_detection_images):
            # publish uncompressed image
            self.pub_img_detections=rospy.Publisher(self.image_detections_topic , Image, queue_size=1)
            # compressed image topic must end in /compressed
            self.pub_img_compressed_detections = rospy.Publisher(self.image_detections_topic+"/compressed", CompressedImage, queue_size=1)

        rospy.loginfo("[MxNet Initialized with model %s]",self.model_name)
        
    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MxNet] %s: %s", param, new_param)
        return new_param


    def enable_cb(self, msg):
        self.start_enabled = msg.data

    def zoom_cb(self, msg):
        # Note: set_zoom is safe, it doesn't take effect until the count of encoded and decoded are equal
        # this was added to the custom functions b/c otherwise could try to decode a pattern that changed when the zoom parameter changed on the fly
        self.zoom_enabled = msg.data
        self.imageprocessor.set_zoom(self.zoom_enabled)

    def encode_detection_msg(self,detections):
        detections_msg = Detection2DArray()
        if (len(detections)>0):
            i=0
            detstring='Object Detected:'
            for det in detections:
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
                detstring=detstring+' '+self.class_names[int(det[0])]+', p=%.2f.'%(det[1])
                i+=1
            rospy.logwarn(detstring)
        self.detection_seq += 1
        return detections_msg

    def report_overlaps(self):
        pct_indices,level0_overlap, level1_xoverlap, level1_yoverlap=self.imageprocessor.get_crop_location_pcts(report_overlaps=True, data_shape=self.data_shape)
        rospy.loginfo("\nReported Overlap\nFor Input Image Shape=%d,%d,%d\nOverlap first-level: %d%%, second-level zoom: %d%% %d%%\n\n", self.data_shape[0],self.data_shape[1],self.data_shape[2], int(100*level0_overlap), int(100*level1_xoverlap), int(100*level1_yoverlap))

    def image_cb(self, image):
        if (not self.reported_overlaps):
            cv2_img = self.bridge.imgmsg_to_cv2(image, "rgb8")
            # get and report the overlap percentages
            self.data_shape=cv2_img.shape
            self.report_overlaps()
            self.reported_overlaps=True

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

                    # get list of crops, encode image into the specified crop pattern if zoom is enabled
                    framelist = self.imageprocessor.encode_crops(frame)

                    # pass all frame crops to classifier, returns a list of detections, or a list of zeros if nothing detected in a crop
                    # e.g. every crop gets a detection array added to the list
                    list_of_crop_detections,num_detections = self.classifier.detect(framelist, self.threshold)

                    # decode the detections list for the encoded crop pattern into original image locations
                    decoded_image_detections = self.imageprocessor.decode_crops(list_of_crop_detections)

                    # if there are no detections, continue
                    if num_detections==0:
                        return

                    # package up the list of detections as a message
                    detections_msg = self.encode_detection_msg(decoded_image_detections)

                    self.pub_detections.publish(detections_msg)

                    # if specified, publish images with bounding boxes if detections present
                    if (self.publish_detection_images):
                        # overlay detections on the frame
                        frame = self.imageprocessor.overlay_detections(frame, decoded_image_detections)
                        try:
                            # send uncompressed image
                            self.pub_img_detections.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
                            # send compressed image
                            msg = CompressedImage()
                            msg.header.stamp = rospy.Time.now()
                            msg.format = "jpeg"
                            msg.data = convert_frame_to_jpeg_string(frame)
                            self.pub_img_compressed_detections.publish(msg)
                        except CvBridgeError as e:
                            print(e)
                    if (self.save_detections):
                        cv2.imwrite(self.save_directory+'/mxnet_detection_%05d.jpg'%(self.detection_seq),frame[:,:,[2,1,0]])

                except CvBridgeError, e:
                    rospy.logerr(e)


if __name__ == '__main__':
    rospy.init_node("mxnet_ssd_node", anonymous=False, log_level=rospy.INFO)
    ros_mxnet_ssd = RosMxNetSSD()
    rospy.spin()

