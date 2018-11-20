#!/usr/bin/env python

import cv2
import time
import os
import numpy as np
from mxnet_ssd import MxNetSSDClassifier
from mxnet_ssd_custom_functions import SSDCropPattern, convert_frame_to_jpeg_string, write_image_detection

# input video, output same with det.mp4
filename='/home/ebeall/rosbags/bagvideo_9_5_2018.mp4'
filename='/home/ebeall/Downloads/ChaseVideo_curie_2018-09-11_20-20-25.mp4'
import sys
print 'operating on '+filename
if (len(sys.argv)!=2):
    print 'Must enter filename (mp4) to operate upon'
filename=sys.argv[1]
if filename.find('.mp4')<0 or not os.path.exists(filename):
    print('Must give a valid filename, you gave: '+filename)

threshold = 0.35
level0_ncrops = 2
level1_xcrops = 3
level1_ycrops = 2
level1_crop_size = 420
zoom_enabled=True
        
# location of mxnet model and name, GPU and number of classes
classes = 'goose, person, golfcart, lawncare, dog'
batch_size=10
#batch_size=1
model_directory = '/home/ebeall/mxnet_ssd'
network = 'custom-ssd_512_resnet50_v1_custom'
model_filename = 'goosenet_v0_2_1.params'
#network = 'custom-yolo3_darknet53_custom'
#model_filename = 'yolo3_darknet53_goose_0040_0.8519.params'

classes = [c.strip(' ') for c in classes.strip('\n').split(',')]
num_classes= len(classes)
enable_gpu=True

# SSD classes for handling data/detections
classifier = MxNetSSDClassifier(model_directory, model_filename, network, batch_size, enable_gpu, num_classes)
imageprocessor = SSDCropPattern(zoom_enabled, level0_ncrops, level1_xcrops, level1_ycrops, level1_crop_size)

cap = cv2.VideoCapture(filename)
num_frames=0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    num_frames=num_frames+1
    if ret == True:
        # get list of crops, encode image into the specified crop pattern if zoom is enabled
        framelist = imageprocessor.encode_crops(frame)

        # pass all frame crops to classifier, returns a list of detections, or a list of zeros if nothing detected in a crop
        # e.g. every crop gets a detection array added to the list
        list_of_crop_detections,num_detections = classifier.detect(framelist, threshold)

        # decode the detections list for the encoded crop pattern into original image locations
        decoded_image_detections = imageprocessor.decode_crops(list_of_crop_detections)

        savefile=filename.replace('.mp4','')+'_'+network+'_detection_%05d.jpg'%(num_frames)

        # if there are detections, overlay
        #cv2.imshow('Original',frame)
        if num_detections>0:
            frame = imageprocessor.overlay_detections(frame, decoded_image_detections)

        write_image_detection(savefile,frame)
        #write_image_detection(savefile,frame[:,:,[2,1,0]])
        cv2.imshow('Processed',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break

print num_frames
cap.release()
cv2.destroyAllWindows()


