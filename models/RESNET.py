import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')


class Identityblock(layers.Layer):
    def __init__(self,level,block,filters):
        super(Identityblock, self).__init__()
        
        # layers will be called conv{level}_iden{block}_{convlayer_number_within_block}'
        self.conv_name = f'conv{level}_block{block}' + '_{layer}_{type}'
        self.new_name = f'conv{level}_block{block}' + '_{type}'
        
        # unpack number of filters to be used for each conv layer
        self.f1, self.f2, self.f3 = filters

        self.conv1 = Conv2D(filters=self.f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=self.conv_name.format(layer=1, type='conv'), kernel_initializer=glorot_uniform(seed=0))
        self.bn1 = BatchNormalization(axis=3, name=self.conv_name.format(layer=1, type='bn'))
        self.act1 = Activation('relu', name=self.conv_name.format(layer=1, type='relu'))

        # second convolutional layer
        self.conv2 = Conv2D(filters=self.f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=self.conv_name.format(layer=2, type='conv'), kernel_initializer=glorot_uniform(seed=0))
        self.bn2 = BatchNormalization(axis=3, name=self.conv_name.format(layer=2, type='bn'))
        self.act2 = Activation('relu', name=self.conv_name.format(layer=2, type='relu'))

        # third convolutional layer
        self.conv3 = Conv2D(filters=self.f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=self.conv_name.format(layer=3, type='conv'), kernel_initializer=glorot_uniform(seed=0))
        self.bn3 = BatchNormalization(axis=3, name=self.conv_name.format(layer=3, type='bn'))

        # add shortcut branch to main path
        self.add = Add(name=self.new_name.format(type='add'))

        # relu activation at the end of the block
        self.act4 = Activation('relu', name=self.new_name.format(type='out'))
        
    def call(self, input):
        x= input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)

        # add shortcut branch to main path
        x = self.add([x, input])

        # relu activation at the end of the block
        x = self.act4(x)
        
        return x
    
    
class Convolutionalblock(layers.Layer):
    def __init__(self,level,block,filters,s=(2,2)):
        super(Convolutionalblock, self).__init__()
        # layers will be called conv{level}_{block}_{convlayer_number_within_block}'
        self.conv_name = f'conv{level}_block{block}' + '_{layer}_{type}'
        self.new_name = f'conv{level}_block{block}' + '_{type}'
        
        # unpack number of filters to be used for each conv layer
        self.f1, self.f2, self.f3 = filters
        
        
        self.conv1 = Conv2D(filters=self.f1, kernel_size=(1, 1), strides=s, padding='valid',
                    name=self.conv_name.format(layer=1, type='conv'),
                    kernel_initializer=glorot_uniform(seed=0))
        
        self.bn1 = BatchNormalization(axis=3, name=self.conv_name.format(layer=1, type='bn'))
        self.act1 = Activation('relu', name=self.conv_name.format(layer=1, type='relu'))
        
        self.conv2=Conv2D(filters=self.f2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    name=self.conv_name.format(layer=2, type='conv'),
                    kernel_initializer=glorot_uniform(seed=0))
        
        self.bn2=BatchNormalization(axis=3, name=self.conv_name.format(layer=2, type='bn'))
        self.act2=Activation('relu', name=self.conv_name.format(layer=2, type='relu'))
        
        self.conv3=Conv2D(filters=self.f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                    name=self.conv_name.format(layer=3, type='conv'),
                    kernel_initializer=glorot_uniform(seed=0))
        
        self.bn3=BatchNormalization(axis=3, name=self.conv_name.format(layer=3, type='bn'))
        
        self.conv_s=Conv2D(filters=self.f3, kernel_size=(1, 1), strides=s, padding='valid',
                                name=self.conv_name.format(layer='0', type='conv'),
                                kernel_initializer=glorot_uniform(seed=0))
        self.bn_s=BatchNormalization(axis=3, name=self.conv_name.format(layer='0', type='bn'))
        
        self.add=Add(name=self.new_name.format(type='add'))
        
        self.act4=Activation('relu', name=self.new_name.format(type='out'))
        
  
    def call(self, input):
        x= input
        # first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # third convolutional layer
        x =self.conv3(x)
        x = self.bn3(x)

        # shortcut path
        x_shortcut = self.conv_s(input)
        x_shortcut = self.bn_s(x_shortcut)

        # add shortcut branch to main path
        x = self.add([x, x_shortcut])

        # nonlinearity
        x = self.act4(x)

        return x
    
    
class Resnet50nch(Model):
    def __init__(self, classes=None ):
        super(Resnet50nch, self).__init__()
        
        # self.output_features = []
        
        self.classes = classes
        
        self.padding = ZeroPadding2D((3, 3),name='conv1_pad')     
        
        self.conv1=Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1),
                name='conv1_conv',
                kernel_initializer=glorot_uniform(seed=0))
        
        self.bn1 = BatchNormalization(axis=3, name='conv1_bn')
        self.act1 = Activation('relu',name='conv1_relu')
        self.padding2 = ZeroPadding2D((1, 1),name='pool1_pad')
        
        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2),name='pool1_pool')
                    
        
        self.convblock21 = Convolutionalblock(level=2, block=1, filters=[64, 64, 256], s=(1, 1))

        # 2x identity blocks
        self.identity22 = Identityblock(level=2, block=2, filters=[64, 64, 256])
        self.identity23 = Identityblock(level=2, block=3, filters=[64, 64, 256])

        ### Level 3 ###

        # 1x convolutional block
        self.convblock31 = Convolutionalblock(level=3, block=1, filters=[128, 128, 512], s=(2, 2))

        # 3x identity blocks
        self.identity32 = Identityblock(level=3, block=2, filters=[128, 128, 512])
        self.identity33 = Identityblock(level=3, block=3, filters=[128, 128, 512])
        self.identity34 = Identityblock(level=3, block=4, filters=[128, 128, 512])

        ### Level 4 ###
        # 1x convolutional block
        self.convblock41 = Convolutionalblock(level=4, block=1, filters=[256, 256, 1024], s=(2, 2))
        # 5x identity blocks
        self.identity42 = Identityblock(level=4, block=2, filters=[256, 256, 1024])
        self.identity43 = Identityblock(level=4, block=3, filters=[256, 256, 1024])
        self.identity44 = Identityblock(level=4, block=4, filters=[256, 256, 1024])
        self.identity45 = Identityblock(level=4, block=5, filters=[256, 256, 1024])
        self.identity46 = Identityblock(level=4, block=6, filters=[256, 256, 1024])

        ### Level 5 ###
        # 1x convolutional block
        self.convblock51 = Convolutionalblock(level=5, block=1, filters=[512, 512, 2048], s=(2, 2))
        # 2x identity blocks
        self.identity52 = Identityblock(level=5, block=2, filters=[512, 512, 2048])
        self.identity53 = Identityblock(level=5, block=3, filters=[512, 512, 2048])
        
        if self.classes is not None:
            self.avgpool = AveragePooling2D(pool_size=(7, 7), name='avg_pool')
            self.flatten = Flatten()
            self.dense = Dense(self.classes, activation='softmax', name='predictions', kernel_initializer=glorot_uniform(seed=0))
            
    
    def load_pretrained_weights(self, channels):
        
        if self.classes is None:
            include_top = False
        else:
            include_top = True
            
        resnet_base=keras.applications.ResNet50(include_top=include_top, weights='imagenet', input_shape=(224, 224, 3))
        # get the names of the layers in the model
        
        layer_names = [layer.name for layer in self.layers]
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            name = layer_names[i]
            
            if layer.get_weights() == []:
                    continue
            if 'block' in name:
                layers=self.get_layer(name)
                for layer in layers._flatten_layers(include_self=False):
                    layer_name=layer.name
                    weights = resnet_base.get_layer(layer_name).get_weights()
                    layer.set_weights(weights)
            if name == 'conv1_conv':
                if channels > 3:
                    weights = resnet_base.get_layer(name).get_weights()
                    additional_weights = np.random.rand(7, 7, channels-3, 64)* 0.01
                    weights[0] = np.concatenate((weights[0], additional_weights), axis=2)
                    layer.set_weights(weights)
                         
            else:
                name= layer.name
                weights = resnet_base.get_layer(name).get_weights()
                layer.set_weights(weights)
                    
                
    def call(self,input):  
        x = input
        out_features = []
        # padding
        x = self.padding(x)

        # convolutional layer, followed by batch normalization and relu activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        out_features.append(x)
        
        x = self.padding2(x)
        ### Level 2 ###

        # max pooling layer to halve the size coming from the previous layer
        x = self.maxpool(x)

        # 1x convolutional block
        x = self.convblock21(x)

        # 2x identity blocks
        x = self.identity22(x)
        x = self.identity23(x)

        out_features.append(x)
        
        ### Level 3 ###

        # 1x convolutional block
        x =  self.convblock31(x) 


        # 3x identity blocks
        x = self.identity32(x) 
        x = self.identity33(x) 
        x = self.identity34(x) 

        out_features.append(x)
        
        ### Level 4 ###
        # 1x convolutional block
        x = self.convblock41(x)
        # 5x identity blocks
        x = self.identity42(x)
        x = self.identity43(x)
        x = self.identity44(x)
        x = self.identity45(x)
        x = self.identity46(x)

        out_features.append(x)
        
        ### Level 5 ###
        # 1x convolutional block
        x = self.convblock51(x) 
        # 2x identity blocks
        x = self.identity52(x)
        x = self.identity53(x)
        
        out_features.append(x)
        
        if self.classes is not None:
            # Pooling layers
            x = self.avgpool(x)

            # Output layer
            x = self.flatten(x)
            x = self.dense(x)

        # return x
        return tuple(out_features)
