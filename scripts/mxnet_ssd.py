#!/usr/bin/env python

import mxnet
import time
import numpy as np
from collections import namedtuple
from symbol.symbol_factory import get_symbol


class MxNetSSDClassifier(object):

    # MXNet-based single-shot detector 
    def __init__(self, model_name, model_epoch, model_directory, network_name='resnet50', batch_size=1, gpu_enabled=True, num_classes=20):
        self.ts_start=int(time.time())

        # setup MXNet detector
        self.epoch = model_epoch
        self.prefix = str(model_directory) + '/' + str(model_name)
        self.num_classes = num_classes
        self.batch = namedtuple('Batch', ['data'])
        self.batch_size=batch_size
        self.network = network_name
        if gpu_enabled:
            self.ctx = mxnet.gpu(0)
        else:
            self.ctx = mxnet.cpu()

        # Create Detector
        self.mean_pixels=(123, 117,104)
        self._mean_pixels = mxnet.nd.array(self.mean_pixels).reshape((3,1,1))
        _, args, auxs = mxnet.model.load_checkpoint(self.prefix, self.epoch)
        symbol = get_symbol(self.network, 512, num_classes=self.num_classes)
        #self.mod = mxnet.mod.Module(symbol, context=self.ctx)
        self.mod = mxnet.mod.Module(symbol, context=self.ctx, label_names=None)
        #mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx, fixed_param_names=net.list_arguments())
        #self.mod.bind(for_training=False, data_shapes=[('data', (self.batch_size, 3, 512, 512))], label_shapes=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (self.batch_size, 3, 512, 512))])
        self.mod.set_params(args, auxs)


    def detect(self, image, threshold):
        dets=[]
        num_detections=0
        # operate on list of images
        if (type(image)!=type(list())):
            image = [image]

        # pack list of images into batch size as appropriate
        num_loops = int(np.ceil(len(image)/float(self.batch_size)))
        for i in range(0,num_loops):
            batch_data = mxnet.nd.zeros((self.batch_size, 3, 512, 512))
            start_ind=i*self.batch_size
            stop_ind=min((i+1)*self.batch_size,len(image))
            for j in range(start_ind,stop_ind):
                image_np = image[j]
                data = mxnet.nd.array(image_np)
                data = mxnet.img.imresize(data, 512, 512)
                data = mxnet.nd.transpose(data, (2,0,1))
                data = data.astype('float32')
                data = data - self._mean_pixels
                batch_data[j-start_ind,:,:,:]=data

            self.mod.forward(self.batch([batch_data]))
            outputs=self.mod.get_outputs()[0].asnumpy()
            delta_time=int(time.time())-self.ts_start

            # loop over batch
            for j in range(0,stop_ind-start_ind):
                output=outputs[j,:,:]
                # get rid of -1 classes
                detections = output[np.where(output[:, 0] >= 0)[0]]

                # get rid of below-thresh detections
                detections = detections[np.where(detections[:, 1] >= threshold)[0]]
                dets.append(detections)
                num_detections=num_detections+detections.shape[0]

        # detections are in form [[cls, prob, xmin, ymin, xmax, ymax]]
        return dets,num_detections

