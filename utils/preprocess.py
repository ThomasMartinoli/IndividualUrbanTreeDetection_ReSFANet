import keras
from keras import ops

def preprocess_RGBN(images):
    R = images[...,0:1]
    N = images[...,3:4]
    ndvi = ops.divide_no_nan((N-R),(N+R))
    ndvi *= 127.5

    bgr = keras.applications.vgg16.preprocess_input(images[:,:,:,:3])

    nir = (images[:,:,:,3:4]-127.5)

    images_out = ops.concatenate([bgr,nir,ndvi],axis=-1)

    return images_out

def preprocess_RGB(images):
    bgr = keras.applications.vgg16.preprocess_input(images[:,:,:,:3])

    return bgr


def get_preprocess(bands):
    """ Preprocess function for the 'bands' attribute of the hdf5 file. """
    from utils.preprocess_6ch import preprocess_6bands
    if bands == 'RGB':
        return preprocess_RGB
    if bands == 'RGBN':
        return preprocess_RGBN
    if bands == 'DB,B,G,R,RE,NIR':
        return preprocess_6bands
    raise ValueError(f'Unknown bands config: {bands}')
