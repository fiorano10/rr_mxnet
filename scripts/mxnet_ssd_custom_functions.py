#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2

class SSDCropPattern():
    def __init__(self, zoom_enabled, level0_ncrops, level1_xcrops, level1_ycrops, level1_crop_size):
        # are we applying the sliding window/zoom crop pattern or simply passing one image to detector?
        self.zoom_enabled=zoom_enabled
        self.zoom_set_tmp=zoom_enabled
        self.zoom_changed=False
        # counter to help us determine if safe to change zoom parameter (the crop decoder must be matched to the encode crop pattern)
        self.encode_decode = 0

        # crop pattern
        self.level0_ncrops=level0_ncrops
        self.level1_xcrops=level1_xcrops
        self.level1_ycrops=level1_ycrops
        self.level1_crop_size=level1_crop_size

        # set data_shape to None, on first encoding, this gets set to the frame size
        self.data_shape=None
        self.COLORS=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(255,255,255)]


    # callback to set zoom parameter internally in a safe manner
    def set_zoom(self,zoom_enabled):
        self.zoom_set_tmp=zoom_enabled
        self.zoom_changed=True


    # draw detections (must have been decoded if using any crop)
    def overlay_detections(self,frame, detectionlist):
        # each detection may be blank or it may be a 2D array of detections
        for detarray in detectionlist:
            if (len(detarray)==0):
                continue
            # else draw detection box with one of 7 unique colors (modulo if number of classes is greater than 7)
            for det in detarray:
                pt1=(int(det[2]*self.data_shape[1]),int(det[3]*self.data_shape[0]))
                pt2=(int(det[4]*self.data_shape[1]),int(det[5]*self.data_shape[0]))
                cv2.rectangle(frame, pt1, pt2, self.COLORS[int(det[0]) % len(self.COLORS)],2)
        return frame

    # helper function to apply an offset (percentage) and crop size (percentage) to elements in a given detection array
    # modify in-place (lists are passed by reference in python
    def adjust_inner_box_level1(self, detection_array,xdim, ydim, min_dim_pct, xoffset, yoffset):
        return detection_array

    def adjust_inner_box(self, detection_array,xdim, ydim, min_dim_pct, offset):
        '''
        print ' ***** adjust_inner_box *****'
        print detection_array
        print detection_array.shape
        print ''
        '''
        if len(detection_array)==0:
            return
        for i in range(0,len(detection_array)):
            if (xdim>ydim):
                detection_array[i,2]=offset + min_dim_pct * detection_array[i,2]
                detection_array[i,4]=offset + min_dim_pct * detection_array[i,4]
            elif (xdim<ydim):
                detection_array[i,3]=offset + min_dim_pct * detection_array[i,3]
                detection_array[i,5]=offset + min_dim_pct * detection_array[i,5]
        '''
        print ' ***** after adjust_inner_box *****'
        print detection_array
        print detection_array.shape
        print ''
        '''

    def decode_crops(self,all_detections):
        if (self.data_shape is None):
            print('Not initialized, can only decode if encoded at least once')
            return []
        # only change the zoom value if we're NOT in between encodings and decodings
        if (self.zoom_changed and self.encode_decode==0):
            self.zoom_changed=False
            self.zoom_enabled=self.zoom_set_tmp
        # decode the crops back to original image
        detectionslist=[]
        ydim,xdim=self.data_shape[0:2]
        min_dim=np.min(self.data_shape[0:2])
        max_dim=np.max(self.data_shape[0:2])
        min_dim_pct=float(min_dim)/max_dim
        print ' ***** decode_crops all dets *****'
        print all_detections
        print len(all_detections)
        print all_detections[0].shape
        print ''
        # First-level crop pattern:
        if (self.level0_ncrops==0):
            # no change if resized
            return all_detections
        elif (self.level0_ncrops==1 or not self.zoom_enabled):
            # center-crop, expand out to full image size
            offset_pct=0.5*(1-min_dim_pct)
            # new x value (xmin or xmax) = offset_pct+min_dim_pct*x
            self.adjust_inner_box(all_detections[0],xdim, ydim, min_dim_pct, offset_pct)
        elif (self.level0_ncrops>1):
            # set offsets on linear spaced pattern from 0 to 1.0-cropsize_pct in percent
            offsets=np.linspace(0,1.0-min_dim_pct, self.level0_ncrops)
            for i in range(0,self.level0_ncrops):
                self.adjust_inner_box(all_detections[i],xdim, ydim, min_dim_pct, offsets[i])

        # Second-level crop pattern:
        if (self.level1_xcrops>0 or self.level1_ycrops>0 and self.zoom_enabled):
            # set centroids on linear spaced pattern from 0 to 1-crop_pct
            xoffsets=np.linspace(0,1.0-float(self.level1_crop_size)/xdim, self.level1_xcrops)
            yoffsets=np.linspace(0,1.0-float(self.level1_crop_size)/ydim, self.level1_ycrops)
            for i in range(0,self.level1_xcrops):
                for j in range(0,self.level1_ycrops):
                    for detection_array in all_detections[j+(i*self.level1_ycrops)]:
                        self.adjust_inner_box_level1(detection_array,xdim, ydim, self.level1_crop_size, xoffsets[i], yoffsets[j])

        #print ' ***** after adjust_inner_box *****'
        #print all_detections
        #print ''

        # decrement the encode_decode back to zero
        self.encode_decode = self.encode_decode - 1
        return all_detections

    def encode_crops(self,frame):
        self.data_shape=frame.shape
        ydim,xdim=self.data_shape[0:2]
        min_dim=np.min(self.data_shape[0:2])
        max_dim=np.max(self.data_shape[0:2])
        # increment the encoded images count for safety in decoding same pattern we encoded with
        self.encode_decode = self.encode_decode + 1

        # making a list of crops
        framelist=[]
        # If zoom is not enabled, just pass the first-level n_crops==0 or 1 (resize to square or center crop)
        if (not self.zoom_enabled):
            if (self.level0_ncrops==0):
                framelist.append(cv2.resize(frame,(min_dim,min_dim)))
            else:
                # center crop to square along the larger dimension (note, if dims are equal, do nothing)
                if (xdim > ydim):
                    framelist.append(frame[:,(max_dim-min_dim)/2:-(max_dim-min_dim)/2,:])
                elif (xdim < ydim):
                    framelist.append(frame[(max_dim-min_dim)/2:-(max_dim-min_dim)/2,:,:])
            return framelist

        # Decision tree for crop-pattern
        # First-level crop pattern:
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
            print half_crop
            print centroids
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
        return framelist

