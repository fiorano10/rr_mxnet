#!/usr/bin/env python

import mxnet
import numpy as np
import gluoncv as gcv
import mxnet as mx
import cv2

# MXNet-based semantic segmentation
# models: fcn_resnet50_ade, psp_resnet50_ade, deeplab_resnet50_ade
# we want to make a mask where we want detection results to be valid (mask out water, sky, treetops, etc), just grass plus some vertical boundary extent
# classes in ADE20k: people=12, sky=2, grass=9, road=6, water=21, rock=34. Generally for the goose detector, we might want only grassy and road-like areas.
# for driveable surface detector, we might instead want known, problematic areas like water, lake, sea, rock, etc
class MxNetSegmentation(object):
    def __init__(self, model_directory, model_filename, network_name='deeplab_resnet50_ade', gpu_enabled=True, image_resize=300, mask_values=[12]):
        # model settings
        self.prefix = str(model_directory) + '/' + str(model_filename)
        self.network_name = network_name
        self.image_resize=image_resize
        self.mask_values=mask_values
        if gpu_enabled:
            self.ctx = mxnet.gpu(0)
        else:
            self.ctx = mxnet.cpu()

        self._mean=(0.485, 0.456, 0.406)
        self._std=(0.229, 0.224, 0.225)

        # Create Detector
        self.net = gcv.model_zoo.get_model(self.network_name, pretrained=True)

    def segment(self, image):
        orig_image_size=image.shape
        # set image_size to that desired by model
        image_size=self.image_resize

        data = mxnet.nd.array(image)
        data = mx.image.imresize(data, self.image_resize, self.image_resize)
        data = mx.nd.image.to_tensor(data)
        data = mx.nd.image.normalize(data, mean=self._mean, std=self._std)
        data = data.expand_dims(0).as_in_context(self.ctx)

        output = self.net.forward(data)
        segmentation = mx.nd.squeeze(mx.nd.argmax(output[0], 1))
        segmentation = cv2.resize(segmentation.asnumpy(), (orig_image_size[1],orig_image_size[0]), interpolation=cv2.INTER_NEAREST)

        # reshape image, seg, mask for indexing on pixels in 1D
        image=image.reshape((-1,3))
        segmentation=segmentation.reshape((-1,))
        mask=np.zeros_like(segmentation)
        # AND the absolute value mask values together
        for cls in self.mask_values:
            inds = np.where(segmentation==abs(cls))[0]
            mask[inds]=255

        # handle negative values
        if (int(abs(cls))==int(-1.0*cls)):
            inds = np.where(mask==0)[0]
            # invert the mask
            mask=0*mask
            mask[inds]=255
        else:
            inds = np.where(mask==255)[0]

        # set overlay to RED
        color=np.zeros_like(image)
        color[:,0]=255
        if len(inds)>0:
            image[inds,:]=cv2.addWeighted(image[inds,:],0.5,color[inds,:],0.5,0)

        # reshape back to original
        image=image.reshape((orig_image_size[0],orig_image_size[1],3))
        segmentation=segmentation.reshape((orig_image_size[0],orig_image_size[1]))
        mask=mask.reshape((orig_image_size[0],orig_image_size[1]))

        # return overlay+image and binary mask (0, 255)
        return image, mask.astype(np.uint8)

def convert_frame_to_jpeg_string(frame):
    return np.array(cv2.imencode('.jpg', frame[:,:,[2,1,0]])[1]).tostring()

