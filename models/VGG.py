from tensorflow.keras import layers, initializers

class BaseConv(layers.Layer):
    def __init__(self, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv,self).__init__()
        self.use_bn = use_bn
        self.conv = layers.Conv2D(out_channels, kernel, strides=stride, padding='same',
                           kernel_initializer=initializers.RandomNormal(stddev=0.01))
        self.bn = layers.BatchNormalization()
        if activation is None:
            self.activation = layers.Activation(activation)
        else:
            self.activation = None

    def call(self,inputs):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
