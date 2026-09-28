""" Compute metrics on test set. """
import argparse
import os
import h5py as h5
import yaml
from utils.evaluate import evaluate, save_prediction
from models import SFANet
from utils.preprocess import get_preprocess
import matplotlib as mpl
import tensorflow as tf
mpl.use('Agg')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--max_distance', type=float, default=20, help='max distance from gt to pred tree (in pixels)')
    parser.add_argument('--dataset', default=None,
                        help='dataset folder with images/<name>.tif: if given, confidence maps and labelled detections are saved in <log>')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Replace "0" with the index of the GPU you want to use
    # List visible devices (should only show the one you specified)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
        # Set memory growth for the specified GPU
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {physical_devices[0]}")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU found.")

    params_path = os.path.join(args.log,'params.yaml')
    if os.path.exists(params_path):
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
            mode = params['mode']
            min_distance = params['min_distance']
            threshold_abs = params['threshold_abs'] if mode == 'abs' else None
            threshold_rel = params['threshold_rel'] if mode == 'rel' else None
    else:
        print(f'warning: params.yaml missing -- using default params')
        min_distance = 3
        threshold_abs = None
        mode='abs'
        threshold_rel = 0.2

    f = h5.File(args.data,'r')
    images = f[f'test/images'][:]
    gts = f[f'test/gt'][:]

    bands = f.attrs['bands']
    print(bands)

    training_model, model = SFANet.build_model(
        images.shape[1:],
        preprocess_fn=get_preprocess(bands))

    weights_path = SFANet.find_weights(args.log)
    print(f'carico i pesi finali: {weights_path}')
    training_model.load_weights(weights_path)

    print('----- getting predictions from trained model -----')
    preds = model.predict(images,verbose=True,batch_size=1)[...,0]

    print('----- calculating metrics -----')
    results = evaluate(
        gts=gts,
        preds=preds,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        max_distance=args.max_distance,
        return_locs=True)

    with open(os.path.join(args.log,'results.txt'),'w') as f_out:
        f_out.write('precision: '+str(results['precision'])+'\n')
        f_out.write('recall: '+str(results['recall'])+'\n')
        f_out.write('fscore: '+str(results['fscore'])+'\n')
        f_out.write('rmse [px]: '+str(results['rmse'])+'\n')

    print('------- results for: ' + args.log + ' ---------')
    if mode == 'rel':
        print('threshold_rel:',threshold_rel)
    else:
        print('threshold_abs:',threshold_abs)
    print('precision: ',results['precision'])
    print('recall: ',results['recall'])
    print('fscore: ',results['fscore'])
    print('rmse [px]: ',results['rmse'])

    if args.dataset is not None:
        save_prediction(f['test/names'][:],args.dataset,args.log,results,preds)

if __name__ == '__main__':
    main()
