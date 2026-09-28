""" Compute average precision on test set. """
import numpy as np
import argparse
import os
import h5py as h5
import yaml
from utils.evaluate import calculate_ap, test_all_thresholds_fast
from models import SFANet
from utils.preprocess import get_preprocess
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--max_distance', type=float, default=20, help='max distance from gt to pred tree (in pixels)')

    args = parser.parse_args()

    f = h5.File(args.data,'r')
    images = f[f'test/images'][:]
    gts = f[f'test/gt'][:]

    # Specify the GPU to use (e.g., "0" is the first GPU, "1" is the second GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Replace with the index of the GPU you want to use
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

    training_model, model = SFANet.build_model(
        images.shape[1:],
        preprocess_fn=get_preprocess(f.attrs['bands']))

    weights_path = SFANet.find_weights(args.log)
    print(f'----- loading weights: {weights_path} -----')
    training_model.load_weights(weights_path)

    print('----- getting predictions from trained model -----')
    preds = model.predict(images,verbose=True,batch_size=1)[...,0]

    print('----- calculating metrics -----')
    thresholds, precisions, recalls = test_all_thresholds_fast(
        gts=gts,
        preds=preds,
        max_distance=args.max_distance)
    ap = calculate_ap(precisions,recalls)

    # ----- sort -----
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    thresholds = thresholds[order]

    # ----- F1 -----
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    best_r = recalls[best_idx]
    best_p = precisions[best_idx]
    best_f1 = f1_scores[best_idx]
    best_th = thresholds[best_idx]

    # ----- find closest threshold to the one found by tune_percentile (params.yaml) -----
    tuned_threshold = None
    params_path = os.path.join(args.log,'params.yaml')
    if os.path.exists(params_path):
        with open(params_path,'r') as pf:
            params = yaml.safe_load(pf)
        if params.get('mode') == 'abs' and params.get('threshold_abs') is not None:
            tuned_threshold = float(params['threshold_abs'])
        else:
            print('warning: params.yaml has no absolute threshold -- tuned point not plotted')
    else:
        print('warning: params.yaml missing -- tuned point not plotted')

    if tuned_threshold is not None:
        user_idx = np.argmin(np.abs(thresholds - tuned_threshold))

        user_r = recalls[user_idx]
        user_p = precisions[user_idx]
        user_f1 = f1_scores[user_idx]
        user_th = thresholds[user_idx]

    # ----- plot -----
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, linewidth=2)

    # best point
    plt.scatter(best_r, best_p, marker='o', zorder=5)
    plt.annotate(
        f'BEST\nF1={best_f1:.3f}\nP={best_p:.3f}, R={best_r:.3f}\nT={best_th:.4f}',
        (best_r, best_p),
        textcoords="offset points",
        xytext=(10,25)
    )

    # tuned threshold point (from tune_percentile)
    if tuned_threshold is not None:
        plt.scatter(user_r, user_p, marker='x', zorder=5)
        plt.annotate(
            f'TUNED\nF1={user_f1:.3f}\nP={user_p:.3f}, R={user_r:.3f}\nT={user_th:.4f}',
            (user_r, user_p),
            textcoords="offset points",
            xytext=(-40,-55)
        )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
    plt.grid()

    # ----- save -----
    out_path = os.path.join(args.log, 'precision_recall_curve.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    with open(os.path.join(args.log,'ap_results.txt'),'w') as f: 
        f.write('average precision: '+str(ap)) 
        print('------- results for: ' + args.log + ' ---------') 
        print('average precision: ',ap)
        if tuned_threshold is not None:
            tuned_line = (f'tuned threshold (params.yaml): {tuned_threshold:.6f} '
                          f'(closest evaluated: {user_th:.6f}) '
                          f'precision={user_p:.4f} recall={user_r:.4f} f1={user_f1:.4f}')
            f.write('\n' + tuned_line)
            print(tuned_line)

if __name__ == '__main__':
    main()
