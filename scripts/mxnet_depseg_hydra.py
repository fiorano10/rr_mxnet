#!/usr/bin/env python

import numpy as np
import mxnet as mx
import cv2
from deeplabv3 import _DeepLabPlusHead


# MXNet-based semantic segmentation and monocular depth estimation, trained on Cityscapes and KITTI data
# Hydra network that has common head (resent18, 34, 50 or 101-based) and separate tails for 
# segmentation (deeplabv3+) and monocular depth (similar to monodepth)
class MxNetDepthSegmentationHydra(object):
    def __init__(self, model_directory, model_filename, resnet_depth=18, seg_classes=10, gpu_enabled=True, image_resize=300):
        # model settings
        self.prefix = str(model_directory) + '/' + str(model_filename)
        # if image_resize is 1D, expand to 2D
        if len(image_resize)==1:
            image_resize=[image_resize,image_resize]
        self.image_resize=image_resize
        self.resnet_depth=resnet_depth
        self.seg_classes=seg_classes
        if gpu_enabled:
            self.ctx = mx.gpu(0)
        else:
            self.ctx = mx.cpu()

        #self.segmenter = MxNetDepthSegmentationHydra(None, None, self.network, self.enable_gpu, image_resize=self.image_resize)
        # Create Detector
        self.net = UResNetHydra(self.ctx, input_shape=image_resize, num_classes=seg_classes, depth=resnet_depth)
        self.net.load_parameters(self.prefix, self.ctx)
        #self.net.collect_params().reset_ctx(self.ctx)

    def segment_and_depth(self, image):
        orig_image_size=image.shape
        print 'received image'

        # set image_size to that desired by model
        image_size=self.image_resize

        data = mx.nd.array(image)
        data = mx.image.imresize(data, self.image_resize[1], self.image_resize[0])
        # normalize images to 0-1 range
        data = mx.nd.image.to_tensor(data)
        data = data.expand_dims(0).as_in_context(self.ctx)

        output = self.net.forward(data)
        segmentation = mx.nd.squeeze(mx.nd.argmax(output[4][0,:,:,:], axis=0))

        # resize to original image size
        segmentation = cv2.resize(segmentation.asnumpy(), (orig_image_size[1],orig_image_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = output[0][0,0,:,:].asnumpy()
        #depth = 1.0/depth
        depth = (255.0*depth/np.max(depth)).astype(np.uint8)
        # invert depth about 1 to turn into approx real depths
        #depth[depth>0] = 1./depth[depth>0]
        # scale up to 255
        #depth=(255.0*depth/np.max(depth)).astype(np.uint8)
        depth = cv2.resize(depth, (orig_image_size[1],orig_image_size[0]), interpolation=cv2.INTER_LINEAR)

        # return segmentation
        return segmentation, depth
    

def convert_frame_to_jpeg_string(frame):
    return np.array(cv2.imencode('.jpg', frame[:,:,[2,1,0]])[1]).tostring()


class upconv(mx.gluon.nn.HybridBlock):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, output_shape=[256,512]):
        super(upconv, self).__init__()
        self.output_shape = output_shape
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def hybrid_forward(self, F, x):
        x = F.UpSampling(x,scale=2,sample_type='nearest')
        x = self.conv1(x)
        return x

class get_disp(mx.gluon.nn.HybridBlock):
    def __init__(self, num_in_layers, name=''):
        super(get_disp, self).__init__()
        self.conv1 = mx.gluon.nn.Conv2D(in_channels=num_in_layers, channels=2, kernel_size=3, strides=1, prefix='dispconv_'+name)
        self.normalize = mx.gluon.nn.BatchNorm()

    def hybrid_forward(self, F, x):
        p = 1
        # x is in form NCHW, pad the HW by p on both sides
        p2d = (0,0,0,0,p, p, p, p)
        x = self.conv1(F.pad(x, pad_width=p2d, mode="constant", constant_value=0))
        x = self.normalize(x)
        return 0.3 * F.sigmoid(x)


class get_seg(mx.gluon.nn.HybridBlock):
    # in_layers ranges from 4,8,16,32 (reverse), and always output num_classes channels
    def __init__(self, num_in_layers, num_classes=20, name=''):
        super(get_seg, self).__init__()
        num_mid_layers=num_in_layers//4
        self.conv1 = mx.gluon.nn.Conv2D(in_channels=num_in_layers, channels=num_mid_layers, kernel_size=3, strides=1, use_bias=False, prefix='segconv1_'+name)
        self.normalize = mx.gluon.nn.BatchNorm()
        self.relu = mx.gluon.nn.Activation(activation='relu')
        self.dropout = mx.gluon.nn.Dropout(0.1)
        self.conv2 = mx.gluon.nn.Conv2D(in_channels=num_mid_layers, channels=num_classes, kernel_size=1, prefix='segconv2_'+name)

    def hybrid_forward(self, F, x):
        p = 1
        p2d = (0,0,0,0,p, p, p, p)
        x = self.conv1(F.pad(x, pad_width=p2d, mode="constant", constant_value=0))
        x = self.normalize(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class conv(mx.gluon.nn.HybridBlock):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = mx.gluon.nn.Conv2D(in_channels=num_in_layers, channels=num_out_layers, kernel_size=kernel_size, strides=stride)
        self.normalize = mx.gluon.nn.BatchNorm()

    def hybrid_forward(self, F, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (0,0,0,0, p, p, p, p)
        x = self.conv_base(F.pad(x, pad_width=p2d, mode="constant", constant_value=0))
        x = self.normalize(x)
        x = F.LeakyReLU(x, act_type='elu')
        return x

class UResNetHydra(mx.gluon.nn.HybridBlock):
    def __init__(self, ctx, input_shape, num_classes=21, depth=50, version=1, pretrained=True, debug=False, segdepth=False):
        super(UResNetHydra, self).__init__()
        self.ctx=ctx
        self.debug=debug
        self.input_shape=input_shape
        self.skip_size = 64
        if (depth<50):
            self.filters = [4, 8, 16, 32, 64, 128, 256, 512]
        elif (depth==50):
            self.filters = [16, 32, 64, 128, 256, 512, 1024, 2048]
        # v2 resnets all start at end_feature==9 and go up to 11 (range(9,12))
        if depth==18:
            if version==1:
                resnet = mx.gluon.model_zoo.vision.resnet18_v1(pretrained=pretrained, ctx=self.ctx)
            else:
                resnet = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=pretrained, ctx=self.ctx)
        elif depth==34:
            if version==1:
                resnet = mx.gluon.model_zoo.vision.resnet34_v1(pretrained=pretrained, ctx=self.ctx)
            else:
                resnet = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=pretrained, ctx=self.ctx)
        elif depth==50:
            if version==1:
                resnet = mx.gluon.model_zoo.vision.resnet50_v1(pretrained=pretrained, ctx=self.ctx)
            else:
                resnet = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=pretrained, ctx=self.ctx)
        else:
            print 'Not defined'
        # ResNet_v2 inspect children, delete the 9th-12th children of net.features
        # children 0-4 are BN, conv2d(7x7), BN, Relu, maxpool
        # children 5,6,7,8 are the stages1-4
        # ResNet_v1 is shorter, no initial batchnorm
        # remove from stage3 onwards
        with resnet.name_scope():
            if version==1:
                del resnet._children['features']._children['8']
            else:
                for i in range(9,13):
                    del resnet._children['features']._children[str(i)]
            del resnet._children['output']._children
            del resnet._children['output']
            del resnet.output
        # reset stride in last two encode blocks to maintain higher resolution
        resnet._children['features']._children['6'][0].body[0]._kwargs['stride']=(1,1)
        resnet._children['features']._children['7'][0].body[0]._kwargs['stride']=(1,1)
        resnet._children['features']._children['6'][0].downsample[0]._kwargs['stride']=(1,1)
        resnet._children['features']._children['7'][0].downsample[0]._kwargs['stride']=(1,1)
        # dilate the 3x3 conv in the first body of the last two encode blocks in body
        # and pad to retain shape. the downsample is same as before so should be like 2-layer pyramid pool
        resnet._children['features']._children['6'][0].body[3]._kwargs['dilate']=(2,2)
        resnet._children['features']._children['6'][0].body[3]._kwargs['pad']=(2,2)
        resnet._children['features']._children['7'][0].body[3]._kwargs['dilate']=(2,2)
        resnet._children['features']._children['7'][0].body[3]._kwargs['pad']=(2,2)

        self.prebn = None
        st=0
        # v2 resnet has an initial batchnorm layer with scale, center==False
        if version==2:
            self.prebn = resnet.features[0]
            st=1
        self.firstconv = resnet.features[st+0]
        self.firstbn = resnet.features[st+1]
        self.firstrelu = resnet.features[st+2]
        self.firstmaxpool = resnet.features[st+3]
        self.secondmaxpool = resnet.features[st+3]
        self.encoder1 = resnet.features[st+4]
        self.encoder2 = resnet.features[st+5]
        self.encoder3 = resnet.features[st+6]
        self.encoder4 = resnet.features[st+7]
        resnet.collect_params().setattr('lr_mult',0.1)

        # if ds3==8, we lose the first two upsamplers...
        ds3=8
        ds4=ds3//2
        ds5=ds4//2

        self.skip3_chred = mx.gluon.nn.Conv2D(in_channels=self.filters[-3], channels=self.filters[-4], kernel_size=1)
        self.skip4_chred = mx.gluon.nn.Conv2D(in_channels=self.filters[-2], channels=self.filters[-3], kernel_size=1)
        self.skip5_chred = mx.gluon.nn.Conv2D(in_channels=self.filters[-1], channels=self.filters[-2], kernel_size=1)

        self.upconv4 = upconv(self.filters[-2]+self.filters[-3], self.filters[-3], 3, [self.input_shape[0]//ds3,self.input_shape[1]//ds3])
        self.iconv4 = conv(self.filters[-3] + self.filters[-4], self.filters[3], 3, 1)
        self.disp4_layer = get_disp(self.filters[3], name='4')

        self.upconv3 = upconv(self.filters[3], self.filters[2], 3, [self.input_shape[0]//ds4,self.input_shape[1]//ds4])
        self.iconv3 = conv(self.skip_size + self.filters[2] + 2 , self.filters[2], 3, 1)
        self.disp3_layer = get_disp(self.filters[2], name='3')

        self.upconv2 = upconv(self.filters[2], self.filters[1], 3, [self.input_shape[0]//ds5,self.input_shape[1]//ds5])
        self.iconv2 = conv(self.skip_size + self.filters[1] + 2 , self.filters[1], 3, 1)
        self.disp2_layer = get_disp(self.filters[1], name='2')

        self.upconv1 = upconv(self.filters[1], self.filters[0] // 2, 3, [self.input_shape[0],self.input_shape[1]])
        self.iconv1 = conv(self.filters[0]//2 + 2 , self.filters[0], 3, 1)
        self.disp1_layer = get_disp(self.filters[0], name='1')

        # for deeplabv3+, leave resnet encoders 3/4 alone
        # just add intermediate decoding layer that borrows from one of the skip layers
        # this is why its so much faster than segnet in gluoncv
        with self.name_scope():
            self.head = _DeepLabPlusHead(num_classes, in_channels=self.filters[-1], input_shape=self.input_shape)
            self.head.initialize(ctx=ctx)

        if self.debug:
            print self

    def hybrid_forward(self, F, data):
        if self.debug:
            print('data input='+str(data.shape))
            print('ensure the input dimension is divisible by 32 in the data_shape')

        if (self.prebn is not None):
            data = self.prebn(data)
        # Initial hard pooling layers
        e0 = self.firstconv(data)
        x = self.firstbn(e0)
        x = self.firstrelu(x)
        mp1 = self.firstmaxpool(x)
        # Encoder
        e1 = self.encoder1(mp1)
        e2 = self.encoder2(e1)
        mp2 = self.secondmaxpool(e2)
        e3 = self.encoder3(mp2)
        e4 = self.encoder4(e3)
        if self.debug:
            print('encoder0=  '+str(e0.shape))
            print('mp1=       '+str(mp1.shape))
            print('encoder1=  '+str(e1.shape))
            print('encoder2=  '+str(e2.shape))
            print('mp2=       '+str(mp2.shape))
            print('encoder3=  '+str(e3.shape))
            print('encoder4=  '+str(e4.shape))

        skip1 = e0
        skip2 = e1
        skip3 = e2
        skip4 = e3
        skip5 = e4
        if self.debug:
            print('skip1(e0)= '+str(skip1.shape))
            print('skip2(e1)= '+str(skip2.shape))
            print('skip3(e2)= '+str(skip3.shape))
            print('skip4(e3)= '+str(skip4.shape))
            print('skip5(e4)= '+str(skip5.shape))

        # canonical segnet uses e4, e3 as inputs to the FCN heads
        self.seg_output = self.head(skip5, skip2)

        if self.debug:
            print('seg_output='+str(self.seg_output.shape))

        # Decoder
        skip3_ch = self.skip3_chred(skip3)
        skip4_ch = self.skip4_chred(skip4)
        skip5_ch = self.skip5_chred(skip5)
        concat5 = F.concat(skip5_ch, skip4_ch, dim=1)
        concat4 = F.concat(self.upconv4(concat5), skip3_ch, dim=1)
        if self.debug:
            print('skip4_ch = '+str(skip4_ch.shape))
            print('skip5_ch = '+str(skip5_ch.shape))
            print('concat5=    '+str(concat5.shape))
            print('upconv4=    '+str(self.upconv4(concat5).shape))
            print('concat4=   '+str(concat4.shape))

        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = F.UpSampling(self.disp4, num_filter=1,scale=2, sample_type='nearest')

        concat3 = F.concat(self.upconv3(iconv4), skip2, self.udisp4, dim=1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = F.UpSampling(self.disp3, num_filter=1,scale=2, sample_type='nearest')

        concat2 = F.concat(self.upconv2(iconv3), skip1, self.udisp3, dim=1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = F.UpSampling(self.disp2, num_filter=1,scale=2, sample_type='nearest')

        concat1 = F.concat(self.upconv1(iconv2), self.udisp2, dim=1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        if self.debug:
            print('iconv4=    '+str(iconv4.shape))
            print('concat3=   '+str(concat3.shape))
            print('iconv3=    '+str(iconv3.shape))
            print('concat2=   '+str(concat2.shape))
            print('iconv2=    '+str(iconv2.shape))
            print('concat1=   '+str(concat1.shape))
            print('iconv1=    '+str(iconv1.shape))

            print('udisp4=    '+str(self.udisp4.shape))
            print('udisp3=    '+str(self.udisp3.shape))
            print('udisp2=    '+str(self.udisp2.shape))

        return [self.disp1, self.disp2, self.disp3, self.disp4, self.seg_output]

