#!/usr/bin/env python

import sys
import os
import numpy as np

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

    def set_zoom(self,zoom_enabled):
        self.zoom_set_tmp=zoom_enabled
        self.zoom_changed=True

    # draw detections (must have been decoded if using any crop)
    def overlay_detections(self,frame, detectionlist):
        if (self.level0_ncrops==0):
            return frame
        print('TODO: not yet implemented, draw detections on image')
        # TODO
        return frame

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
        # TODO

        # decrement the encode_decode back to zero
        self.encode_decode = self.encode_decode - 1
        return detectionslist

    def crop(self,frame):
                    self.data_shape=frame.shape
                    ydim,xdim=self.data_shape[0:2]
                    min_dim=np.min(self.data_shape[0:2])
                    max_dim=np.max(self.data_shape[0:2])
                    # increment the encoded images count for safety in decoding same pattern we encoded with
                    self.encode_decode = self.encode_decode + 1
                    framelist=[]
                    # if zoom is not enabled, just pass the first-level n_crops==0 or 1 (resize to square or center crop)
                    if (not self.zoom_enabled):
                        framelist.append(cv2.resize(frame,(min_dim,min_dim)))

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
                    return framelist

