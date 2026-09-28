import keras
from keras import ops

def preprocess_6bands(images):
    # Estrai RGB (ordine corretto)
    R = images[..., 3:4]
    G = images[..., 2:3]
    B = images[..., 1:2]

    rgb = ops.concatenate([R, G, B], axis=-1)

    # Preprocessing ImageNet (→ BGR + mean subtraction)
    bgr = keras.applications.vgg16.preprocess_input(rgb)

    # Altri canali (decidi una normalizzazione sensata)
    db  = images[..., 0:1] - 127.5
    re  = images[..., 4:5] - 127.5
    nir = images[..., 5:6] - 127.5

    # Output finale: 6 canali
    images_out = ops.concatenate([bgr, db, re, nir], axis=-1)

    return images_out
