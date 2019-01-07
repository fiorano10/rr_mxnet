# pylint: disable=unused-argument
"""Pyramid Scene Parsing Network"""
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
import os
from mxnet import gluon
import mxnet as mx
# pylint: disable-all

class _FCNHead(HybridBlock):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm, norm_kwargs={}):
        super(_FCNHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            inter_channels = in_channels // 4
            with self.block.name_scope():
                self.block.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels,
                                         kernel_size=3, padding=1, use_bias=False))
                self.block.add(norm_layer(in_channels=inter_channels, **norm_kwargs))
                self.block.add(nn.Activation('relu'))
                self.block.add(nn.Dropout(0.1))
                self.block.add(nn.Conv2D(in_channels=inter_channels, channels=channels,
                                         kernel_size=1))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        return self.block(x)

class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, in_channels=2048, norm_layer=nn.BatchNorm, norm_kwargs={}, input_shape=(256,512), **kwargs):
        super(_DeepLabHead, self).__init__()
        self.input_shape=input_shape
        mid_channels=256 #in_channels//8
        with self.name_scope():
            # for CityScapes data, use 6,12,18 and 513 square image crops
            # original deeplab script from gluoncv uses 12,24,36
            #self.aspp = _ASPP(in_channels=in_channels, atrous_rates=[6, 12, 18], norm_layer=norm_layer,
            self.aspp = _ASPP(in_channels=in_channels, atrous_rates=[12, 24, 36], norm_layer=norm_layer,
                              norm_kwargs=norm_kwargs, input_shape=input_shape, **kwargs)
            self.block = nn.HybridSequential()
            # change to depthwise separable
            self.block.add(nn.Conv2D(in_channels=mid_channels, channels=mid_channels,
                                     kernel_size=1, use_bias=False))
            self.block.add(nn.Conv2D(in_channels=mid_channels, channels=mid_channels, groups=mid_channels,
                                     kernel_size=3, padding=1, use_bias=False))
            #self.block.add(nn.Conv2D(in_channels=mid_channels, channels=mid_channels,
            #                         kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=mid_channels, **norm_kwargs))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=mid_channels, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.aspp(x)
        x=self.block(x)
        return x
        #return self.block(x)

class _DeepLabPlusHead(HybridBlock):
    def __init__(self, nclass, in_channels=512, norm_layer=nn.BatchNorm, norm_kwargs={}, input_shape=(256,512), **kwargs):
        super(_DeepLabPlusHead, self).__init__()
        self.input_shape=input_shape
        mid_channels=in_channels//2
        skip_in_channels=64
        skip_out_channels=skip_in_channels//2
        mid_mix_channels=mid_channels+skip_out_channels
        self.quartersteps=True
        with self.name_scope():
            # for CityScapes data, use 6,12,18 and 513 square image crops
            # original deeplab script from gluoncv uses 12,24,36
            #self.aspp = _ASPP(in_channels=in_channels, atrous_rates=[6, 12, 18], norm_layer=norm_layer,
            self.aspp = _ASPP(in_channels=in_channels, atrous_rates=[12, 24, 36], norm_layer=norm_layer,
                              norm_kwargs=norm_kwargs, input_shape=input_shape, **kwargs)
            
            self.aspp_postproc = nn.HybridSequential()
            self.aspp_postproc.add(nn.Conv2D(in_channels=mid_channels, channels=mid_channels,
                                     kernel_size=1))

            self.skip_preproc = nn.HybridSequential()
            self.skip_preproc.add(nn.Conv2D(in_channels=skip_in_channels, channels=skip_out_channels,
                                    kernel_size=1))
            self.skip_preproc.add(norm_layer(in_channels=skip_out_channels, **norm_kwargs))
            self.skip_preproc.add(nn.Activation('relu'))

            self.block = nn.HybridSequential()
            self.block.add(nn.Conv2D(in_channels=mid_mix_channels, channels=mid_channels,
                                     kernel_size=1, use_bias=False))
            self.block.add(nn.Conv2D(in_channels=mid_channels, channels=mid_channels, groups=mid_channels,
                                     kernel_size=3, padding=1, use_bias=False))
            #self.block.add(nn.Conv2D(in_channels=mid_mix_channels, channels=mid_channels,
            #                         kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=mid_channels, **norm_kwargs))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=mid_channels, channels=nclass,
                                     kernel_size=1))

            self.postproc = nn.HybridSequential()
            self.postproc.add(nn.Conv2D(in_channels=mid_mix_channels, channels=mid_mix_channels,
                                     kernel_size=1, use_bias=False))
            self.postproc.add(nn.Conv2D(in_channels=mid_mix_channels, channels=mid_mix_channels, groups=mid_mix_channels,
                                     kernel_size=3, padding=1, use_bias=False))
            #self.postproc.add(nn.Conv2D(in_channels=mid_mix_channels, channels=mid_mix_channels,
            #                        kernel_size=3, padding=1))
            self.postproc.add(norm_layer(in_channels=mid_mix_channels, **norm_kwargs))
            self.postproc.add(nn.Activation('relu'))


    def hybrid_forward(self, F, x, skip):
        #print 'x.shape='+str(x.shape)
        #print 'skip.shape='+str(skip.shape)
        x = self.aspp(x)
        #print 'x.shape='+str(x.shape)
        x = self.aspp_postproc(x)
        #print 'xpp.shape='+str(x.shape)
        # upsample ASPP by factor 4x (assuming we don't first produce the seg output)
        if self.quartersteps:
            upx = F.contrib.BilinearResize2D(x, height=int(self.input_shape[0]//4), width=int(self.input_shape[1]//4))
        else:
            upx = F.contrib.BilinearResize2D(x, height=int(self.input_shape[0]//8), width=int(self.input_shape[1]//8))
        #print 'upx.shape='+str(upx.shape)
        # concat earlier feature layer in here (after reduce its channels to ~32 or 48)
        y = self.skip_preproc(skip)
        if self.quartersteps:
            y = F.contrib.BilinearResize2D(y, height=int(self.input_shape[0]//4), width=int(self.input_shape[1]//4))
        x = F.concat(upx,y,dim=1)
        #print 'y.shape='+str(y.shape)
        #print 'concat shape='+str(x.shape)
        x=self.postproc(x)
        #print 'postproc shape='+str(x.shape)
        x=self.block(x)
        #print 'block shape='+str(x.shape)
        x = F.contrib.BilinearResize2D(x, height=int(self.input_shape[0]), width=int(self.input_shape[1]))
        #print 'final shape='+str(x.shape)
        return x

def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                     kernel_size=1, use_bias=False))
        block.add(nn.Conv2D(in_channels=out_channels, channels=out_channels, groups=out_channels,
                                     kernel_size=3, padding=atrous_rate, dilation=atrous_rate, use_bias=False))
        #block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
        #                    kernel_size=3, padding=atrous_rate,
        #                    dilation=atrous_rate, use_bias=False))
        block.add(norm_layer(in_channels=out_channels, **norm_kwargs))
        block.add(nn.Activation('relu'))
    return block

class _AsppPooling(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, input_shape):
        super(_AsppPooling, self).__init__()
        self.gap = nn.HybridSequential()
        self.input_shape=input_shape
        with self.gap.name_scope():
            self.gap.add(nn.GlobalAvgPool2D())
            self.gap.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))
            self.gap.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            self.gap.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        pool = self.gap(x)
        #print 'AsppPooling '+str(x.shape)
        #return F.contrib.BilinearResize2D(pool, height=self.input_shape[0]//64, width=self.input_shape[1]//64)
        return F.contrib.BilinearResize2D(pool, height=self.input_shape[0]//16, width=self.input_shape[1]//16)

class _ASPP(nn.HybridBlock):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, input_shape):
        super(_ASPP, self).__init__()
        self.input_shape=input_shape
        out_channels = 256
        b0 = nn.HybridSequential()
        with b0.name_scope():
            b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                             kernel_size=1, use_bias=False))
            b0.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
                          norm_kwargs=norm_kwargs, input_shape=input_shape)
        self.b0=b0
        self.b1=b1
        self.b2=b2
        self.b3=b3
        self.b4=b4

        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        with self.concurent.name_scope():
            self.concurent.add(b0)
            self.concurent.add(b1)
            self.concurent.add(b2)
            self.concurent.add(b3)
            self.concurent.add(b4)

        self.project = nn.HybridSequential()
        with self.project.name_scope():
            self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
                                       kernel_size=1, use_bias=False))
            self.project.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            self.project.add(nn.Activation("relu"))
            self.project.add(nn.Dropout(0.5))

    def hybrid_forward(self, F, x):
        return self.project(self.concurent(x))
    '''

        print x.shape
        y0=self.b0(x)
        print y0.shape
        y1=self.b1(x)
        print y1.shape
        y2=self.b2(x)
        print y2.shape
        y3=self.b3(x)
        print y3.shape
        y4=self.b4(x)
        print y4.shape
        x=self.project(mx.ndarray.concat(y0,y1,y2,y3,y4,dim=1))
        print x.shape
        return x
    '''

