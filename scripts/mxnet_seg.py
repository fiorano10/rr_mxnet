#!/usr/bin/env python

import mxnet
import numpy as np
import gluoncv as gcv
import mxnet as mx
from gluoncv.utils.viz import get_color_pallete
import cv2

# MXNet-based semantic segmentation
# models: fcn_resnet50_ade, psp_resnet50_ade, deeplab_resnet50_ade
# we want to make a mask where we want detection results to be valid (mask out water, sky, treetops, etc), just grass plus some vertical boundary extent
# classes in ADE20k: people=12, sky=2, grass=9, road=6, water=21, rock=34. Generally for the goose detector, we might want only grassy and road-like areas.
# for driveable surface detector, we might instead want known, problematic areas like water, lake, sea, rock, etc
class MxNetSegmentation(object):
    def __init__(self, model_directory, model_filename, network_name='deeplab_resnet50_ade', batch_size=1, gpu_enabled=True, image_resize=300, classes=[12]):
        # model settings
        self.prefix = str(model_directory) + '/' + str(model_filename)
        self.batch_size=batch_size
        self.network_name = network_name
        self.image_resize=image_resize
        self.classes=classes
        if gpu_enabled:
            self.ctx = mxnet.gpu(0)
        else:
            self.ctx = mxnet.cpu()

        self._mean=(0.485, 0.456, 0.406)
        self._std=(0.229, 0.224, 0.225)

        # Create Detector
        self.net = gcv.model_zoo.get_model(self.network_name, pretrained=True)
        #else:
        #    print('Not supported')
        #    return
        if (self.batch_size>1):
            print('Not supported')
            return

        self.batch_data=None

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
        #mask = get_color_pallete(segmentation, 'ade20k')

        # overlay segmentation of selected class numbers as ones
        image=image.reshape((-1,3))
        segmentation=segmentation.reshape((-1,))
        color=np.zeros_like(image)
        color[:,0]=255
        for cls in self.classes:
            inds = np.where(segmentation==cls)[0]
            image[inds,:]=cv2.addWeighted(image[inds,:],0.5,color[inds,:],0.5,0)
        image=image.reshape((orig_image_size[0],orig_image_size[1],3))
        segmentation=segmentation.reshape((orig_image_size[0],orig_image_size[1]))

        return image, segmentation

def convert_frame_to_jpeg_string(frame):
    return np.array(cv2.imencode('.jpg', frame[:,:,[2,1,0]])[1]).tostring()

