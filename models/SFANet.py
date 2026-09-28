import os
from tensorflow.keras import Model, layers
from .VGG import  BaseConv

from .RESNET import Resnet50nch

from keras import ops

class BackEnd(Model):
    def __init__(self,half_res=False):
        super(BackEnd,self).__init__()
        self.half_res = half_res

        self.upsample = layers.UpSampling2D(2,interpolation='bilinear')
        self.conv1 = BaseConv(256, 1, 1, activation='relu', use_bn=True)
        self.conv2 = BaseConv(256, 3, 1, activation='relu', use_bn=True)

        self.conv3 = BaseConv(128, 1, 1, activation='relu', use_bn=True)
        self.conv4 = BaseConv(128, 3, 1, activation='relu', use_bn=True)

        self.conv5 = BaseConv(64, 1, 1, activation='relu', use_bn=True)
        self.conv6 = BaseConv(64, 3, 1, activation='relu', use_bn=True)
        self.conv7 = BaseConv(32, 3, 1, activation='relu', use_bn=True)

        if not self.half_res:
            self.conv8 = BaseConv(32, 1, 1, activation='relu', use_bn=True)
            self.conv9 = BaseConv(32, 3, 1, activation='relu', use_bn=True)
            self.conv10 = BaseConv(32, 3, 1, activation='relu', use_bn=True)
    
    def call(self,inputs):
        if self.half_res:
            conv2_2, conv3_3, conv4_3, conv5_3 = inputs
        else:
            conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 = inputs

        x = self.upsample(conv5_3)

        x = ops.concatenate([x, conv4_3], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        x = ops.concatenate([x, conv3_3], axis=-1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.upsample(x)

        x = ops.concatenate([x, conv2_2], axis=-1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        if not self.half_res:
            x = self.upsample(x)
            x = ops.concatenate([x, conv1_2], axis=-1)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)

        return x

class SFANet(Model):
    def __init__(self,half_res=True):
        super(SFANet,self).__init__()
        self.resnet=Resnet50nch()
        self.amp = BackEnd(half_res=half_res)
        self.dmp = BackEnd(half_res=half_res)
        
        self.conv_att = BaseConv(1, 1, 1, activation='sigmoid', use_bn=True)
        self.conv_out = BaseConv(1, 1, 1, activation=None, use_bn=False)
    
    def call(self,inputs):
        x = inputs
        x = self.resnet(x)
        amp_out = self.amp(x)
        dmp_out = self.dmp(x)
        
        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)
        
        return dmp_out, amp_out

def build_model(input_shape,preprocess_fn=None,bce_loss_weight=0.1,half_res=False):
    image = layers.Input(input_shape)
    
    image_preprocessed = preprocess_fn(image)

    sfanet = SFANet(half_res=half_res)
    dmp, amp = sfanet(image_preprocessed)
    outputs = [dmp,amp]
    print('carico i pesi resnet')
    sfanet.resnet.load_pretrained_weights(image_preprocessed.shape[-1])
    print('caricati i pesi resnet')

    training_model = Model(inputs=image,outputs=outputs)
    testing_model = Model(inputs=image,outputs=dmp)
    
    return training_model, testing_model

def find_weights(log,name='best'):
    """ Path of the <name> weights in the log directory: <name>.weights.h5 (Keras 3),
        or weights.<name>.h5 for models trained with the previous version of train.py. """
    path = os.path.join(log,f'{name}.weights.h5')
    legacy_path = os.path.join(log,f'weights.{name}.h5')
    if not os.path.exists(path) and os.path.exists(legacy_path):
        return legacy_path
    return path
