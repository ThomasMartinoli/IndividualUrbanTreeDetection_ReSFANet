import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

from models import SFANet
from utils.preprocess import get_preprocess

import argparse
import os
import shutil

import h5py as h5

def generator(f,batch_size):
    train_images = f['train/images']
    train_confidence = f['train/confidence']
    train_attention = f['train/attention']
    
    inds = np.arange(len(train_images))
    np.random.shuffle(inds)
    idx = 0
    while True:
        batch_inds = inds[idx:idx+batch_size]
        batch_images = np.stack([train_images[i] for i in batch_inds])
        batch_confidence = np.stack([train_confidence[i] for i in batch_inds])
        batch_attention = np.stack([train_attention[i] for i in batch_inds])
        yield batch_images, (batch_confidence, batch_attention)
        idx += batch_size
        if idx >= len(inds):
            np.random.shuffle(inds)
            idx = 0

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to training data hdf5 file')
    parser.add_argument('log', help='path to log directory')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    args = parser.parse_args()

    # Specify the GPU to use (e.g., "0" is the first GPU, "1" is the second GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace with the index of the GPU you want to use
    #List visible devices (should only show the one you specified)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
        #Set memory growth for the specified GPU
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {physical_devices[0]}")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU found.")

    f = h5.File(args.data,'r')
    bands = f.attrs['bands']
    val_images = f['val/images'][:]
    val_confidence = f['val/confidence'][:]
    val_attention = f['val/attention'][:]
    
    preprocess_fn = get_preprocess(bands)

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model, testing_model = SFANet.build_model(
            val_images.shape[1:],
            preprocess_fn=preprocess_fn)
        opt = Adam(args.lr)
        model.compile(optimizer=opt, loss=['mse','binary_crossentropy'], loss_weights=[1,0.1])
    
    
    model.summary()
    
    os.makedirs(args.log,exist_ok=True)

    callbacks = []

    # Keras 3: with save_weights_only the file name must end in .weights.h5
    weights_path = os.path.join(args.log, 'best.weights.h5')
    callbacks.append(ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=True,
            save_best_only=True,
            save_weights_only=True,
            ))
    weights_path = os.path.join(args.log, 'latest.weights.h5')
    callbacks.append(ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=True,
            save_best_only=False,
            save_weights_only=True,
            ))
    tensorboard_path = os.path.join(args.log,'tensorboard')
    shutil.rmtree(tensorboard_path, ignore_errors=True)
    callbacks.append(tf.keras.callbacks.TensorBoard(tensorboard_path))

    gen = generator(f,args.batch_size)
    y_val = (val_confidence, val_attention)

    # the generator already yields batches of args.batch_size;
    # use_multiprocessing no longer exists in Keras 3
    model.fit(
            gen,
            validation_data=(val_images,y_val),
            validation_batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=len(f['train/images'])//args.batch_size+1,
            verbose=True,
            callbacks=callbacks)

if __name__ == '__main__':
    main()
