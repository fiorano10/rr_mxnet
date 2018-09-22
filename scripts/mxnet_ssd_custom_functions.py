#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2
import itertools

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
        if len(detection_array)==0:
            return
        for i in range(0,len(detection_array)):
            if (xdim>ydim):
                detection_array[i,2]=offset + min_dim_pct * detection_array[i,2]
                detection_array[i,4]=offset + min_dim_pct * detection_array[i,4]
            elif (xdim<ydim):
                detection_array[i,3]=offset + min_dim_pct * detection_array[i,3]
                detection_array[i,5]=offset + min_dim_pct * detection_array[i,5]

    def decode_crops(self,all_detections):
        if (self.data_shape is None):
            print('Not initialized, can only decode if encoded at least once')
            return []
        # only change the zoom value if we're NOT in between encodings and decodings
        if (self.zoom_changed and self.encode_decode==0):
            self.zoom_changed=False
            self.zoom_enabled=self.zoom_set_tmp

        # decode the crops back to original image
        pct_indices=self.get_crop_location_pcts()
        if (len(all_detections) != len(pct_indices)):
            print('WARNING, crop pattern (len='+str(len(pct_indices))+') does not match detections (len='+str(len(all_detections))+')')
        ydim,xdim=self.data_shape[0:2]
        for i in range(0,len(pct_indices)):
            xc,yc,w,h=pct_indices[i,:]
            x1=max(xc-w/2,0.0)
            y1=max(yc-h/2,0.0)
            # rebox the detections into the innerbox xc[i]-w[i]/2.0:xc[i]+w[i]/2.0
            for j in range(0,len(all_detections[i])):
                all_detections[i][j,2]=all_detections[i][j,2]*w + x1
                all_detections[i][j,3]=all_detections[i][j,2]*h + y1
                all_detections[i][j,4]=all_detections[i][j,4]*w + x1
                all_detections[i][j,5]=all_detections[i][j,5]*h + y1


        # decrement the encode_decode back to zero
        self.encode_decode = self.encode_decode - 1
        return all_detections

    # create lists containing the box centers and the box sizes in pcts = xc,yc,w,h
    def get_crop_location_pcts(self):
        if (self.level0_ncrops==0):
            return [0.0],[0.0],[1.0],[1.0]
        ydim,xdim=self.data_shape[0:2]
        min_dim=np.min(self.data_shape[0:2])
        max_dim=np.max(self.data_shape[0:2])
        # First-level crop pattern
        xcrop_pct = float(min_dim)/xdim
        ycrop_pct = float(min_dim)/ydim
        xc0=np.linspace(0.5*xcrop_pct,1.0-0.5*xcrop_pct, self.level0_ncrops)
        yc0=np.linspace(0.5*ycrop_pct,1.0-0.5*ycrop_pct, self.level0_ncrops)
        w0=np.asarray([xcrop_pct]*len(xc0))
        h0=np.asarray([ycrop_pct]*len(yc0))
        pct_indices0=np.asarray([[a,b,c,d] for a,b,c,d in zip(xc0,yc0,w0,h0)])
        # determine the overlap percentages

        # Second-level crop pattern - if xcrops or ycrops are 0, then those lists will be zero length
        xcrop_pct = float(self.level1_crop_size)/xdim
        ycrop_pct = float(self.level1_crop_size)/ydim
        xc=np.linspace(0.5*xcrop_pct,1.0-0.5*xcrop_pct, self.level1_xcrops)
        yc=np.linspace(0.5*ycrop_pct,1.0-0.5*ycrop_pct, self.level1_ycrops)
        w=np.asarray([xcrop_pct]*len(xc))
        h=np.asarray([ycrop_pct]*len(yc))
        # these have to be replicated combinatorically and zipped so we have the total possibilities of xc,yc and same for w,h
        pct_indices1=np.asarray([[i[0][0],i[0][1],i[1][0],i[1][1]] for i in zip(list(itertools.product(xc,yc, repeat=1)),list(itertools.product(w,h, repeat=1)))])

        # stack the levels if zoom is set and non-zero length of crop lists
        if (self.zoom_enabled and len(xc)>0 and len(yc)>0):
            pct_indices0=np.vstack((pct_indices0,pct_indices1))

        return pct_indices0

    def encode_crops(self,frame):
        self.data_shape=frame.shape
        ydim,xdim=self.data_shape[0:2]
        min_dim=np.min(self.data_shape[0:2])
        max_dim=np.max(self.data_shape[0:2])
        # increment the encoded images count for safety in decoding same pattern we encoded with
        self.encode_decode = self.encode_decode + 1

        # making a list of crops
        framelist=[]
        # If level0_ncrops==0, resize to square and return
        if (self.level0_ncrops==0):
            framelist.append(cv2.resize(frame,(min_dim,min_dim)))
            return framelist

        # otherwise, loop over the crop indices and add to framelist
        pct_indices=self.get_crop_location_pcts()
        ydim,xdim=self.data_shape[0:2]
        for i in range(0,len(pct_indices)):
            xc,yc,w,h=pct_indices[i,:]
            x1=max(int((xc-w/2)*xdim),0)
            y1=max(int((yc-h/2)*ydim),0)
            x2=min(int((xc+w/2)*xdim),xdim)
            y2=min(int((yc+h/2)*ydim),ydim)
            framelist.append(np.copy(frame[y1:y2,x1:x2,:]))

        return framelist

