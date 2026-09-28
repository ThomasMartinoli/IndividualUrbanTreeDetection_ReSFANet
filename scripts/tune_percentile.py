""" Determine the best absolute detection threshold in one shot (TensorFlow).

Pipeline (all on the *validation* set):
  1. load the trained weights (best.weights.h5, or legacy weights.best.h5) and build the model
  2. predict the confidence / density maps on val/images
  3. split the maps in two groups: images that contain trees (they have an
     associated GT -> gt.sum() > 0) and images without trees
  4. compute the value distribution and the requested percentiles for each
     group  -> these are the candidate thresholds
  5. run the detection (peak_local_max + matching) on the whole validation set
     for every candidate threshold and keep the one with the best F-score
  6. dump percentiles.json / results_by_threshold.json / params.yaml in the log
     directory and print the threshold to use on the test set.

Workflow: prepare_6ch -> train -> tune_percentile -> calculate_ap -> test
"""

import argparse
import json
import os
from datetime import datetime

import h5py as h5
import numpy as np
import tensorflow as tf
import yaml

from models import SFANet
from utils.evaluate import evaluate
from utils.preprocess import get_preprocess


def setup_gpu(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {physical_devices[0]}")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU found.")


def group_percentiles(values, percentiles, clip_min, clip_max):
    """ values: list of 2D arrays. Returns {p: threshold} and some stats. """
    flat = np.concatenate([np.clip(v, clip_min, clip_max).ravel() for v in values])
    table = {int(p): float(np.percentile(flat, p)) for p in percentiles}
    stats = {
        'n_images': len(values),
        'n_pixels': int(flat.size),
        'min': float(flat.min()),
        'max': float(flat.max()),
        'mean': float(flat.mean()),
        'std': float(flat.std()),
        'median': float(np.median(flat)),
    }
    return table, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory (trained weights + outputs)')
    parser.add_argument('--split', default='val', help='hdf5 split used for tuning (default: val)')
    parser.add_argument('--percentiles', default='90,92,95,97,99',
                        help='comma separated percentiles (default: 90,92,95,97,99)')
    parser.add_argument('--min_distance', type=int, default=3,
                        help='min distance between detected peaks (default: 3)')
    parser.add_argument('--max_distance', type=float, default=20,
                        help='max distance from gt to pred tree, in pixels (default: 20)')
    parser.add_argument('--clip_min', type=float, default=0.0,
                        help='lower clip applied to the maps before percentiles (default: 0)')
    parser.add_argument('--clip_max', type=float, default=1.0,
                        help='upper clip applied to the maps before percentiles (default: 1)')
    parser.add_argument('--metric', default='fscore', choices=['fscore', 'precision', 'recall'],
                        help='metric maximised to pick the best threshold (default: fscore)')
    parser.add_argument('--eval_group', default='all', choices=['all', 'tree'],
                        help='which val images are used to score the candidates (default: all)')
    parser.add_argument('--gpu', default='1', help='CUDA_VISIBLE_DEVICES value (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch size (default: 1)')
    parser.add_argument('--weights', default=None,
                        help='weights filename inside the log dir (default: best.weights.h5, or legacy weights.best.h5)')
    parser.add_argument('--cache_preds', action='store_true',
                        help='save/reuse <log>/<split>_preds.npy')
    args = parser.parse_args()

    print('-' * 60)
    print("Ora di esecuzione:", datetime.now())
    print('-' * 60)

    setup_gpu(args.gpu)

    percentiles = [float(p) for p in args.percentiles.split(',') if p.strip() != '']

    # ------------------------------------------------------------------ data
    f = h5.File(args.data, 'r')
    images = f[f'{args.split}/images'][:]
    gts = f[f'{args.split}/gt'][:]
    bands = f.attrs['bands']
    print(f'split={args.split}  images={images.shape}  bands={bands}')

    # ---------------------------------------------------------------- predict
    preds_path = os.path.join(args.log, f'{args.split}_preds.npy')
    if args.cache_preds and os.path.exists(preds_path):
        print('----- loading predictions from file -----')
        preds = np.load(preds_path)
    else:
        preprocess = get_preprocess(bands)
        training_model, model = SFANet.build_model(
            images.shape[1:],
            preprocess_fn=preprocess)

        if args.weights is not None:
            weights_path = os.path.join(args.log, args.weights)
        else:
            weights_path = SFANet.find_weights(args.log)
        print(f'----- loading weights: {weights_path} -----')
        training_model.load_weights(weights_path)

        print('----- getting predictions from trained model -----')
        preds = model.predict(images, verbose=True, batch_size=args.batch_size)[..., 0]
        if args.cache_preds:
            np.save(preds_path, preds)

    # ------------------------------------------------------- tree / no tree
    has_tree = np.array([g.sum() > 0 for g in gts])
    tree_maps = [preds[i] for i in np.where(has_tree)[0]]
    no_tree_maps = [preds[i] for i in np.where(~has_tree)[0]]
    print(f'validation images with trees: {len(tree_maps)}  without trees: {len(no_tree_maps)}')

    percentile_report = {
        'split': args.split,
        'percentiles': percentiles,
        'clip': [args.clip_min, args.clip_max],
    }
    candidates = {}  # threshold value -> label

    if tree_maps:
        tree_table, tree_stats = group_percentiles(tree_maps, percentiles, args.clip_min, args.clip_max)
        percentile_report['tree'] = {'stats': tree_stats, 'percentiles': tree_table}
        print('TREE   percentiles:', tree_table)
        for p, v in tree_table.items():
            candidates.setdefault(v, f'tree_p{p}')
    else:
        print('WARNING: no validation image with trees')

    if no_tree_maps:
        nt_table, nt_stats = group_percentiles(no_tree_maps, percentiles, args.clip_min, args.clip_max)
        percentile_report['no_tree'] = {'stats': nt_stats, 'percentiles': nt_table}
        print('NOTREE percentiles:', nt_table)
        for p, v in nt_table.items():
            candidates.setdefault(v, f'no_tree_p{p}')
    else:
        print('WARNING: no validation image without trees')

    if not candidates:
        raise RuntimeError('no candidate thresholds could be computed')

    # ------------------------------------------------ score every candidate
    if args.eval_group == 'tree':
        eval_idx = np.where(has_tree)[0]
    else:
        eval_idx = np.arange(len(preds))
    eval_gts = gts[eval_idx]
    eval_preds = preds[eval_idx]
    print(f'----- scoring {len(candidates)} candidate thresholds on '
          f'{len(eval_idx)} "{args.eval_group}" images -----')

    results_by_threshold = {}
    best = None
    for thr in sorted(candidates):
        res = evaluate(
            gts=eval_gts,
            preds=eval_preds,
            min_distance=args.min_distance,
            threshold_rel=None,
            threshold_abs=thr,
            max_distance=args.max_distance,
            return_locs=False)
        row = {
            'source': candidates[thr],
            'precision': res['precision'],
            'recall': res['recall'],
            'fscore': res['fscore'],
        }
        results_by_threshold[f'{thr:.8f}'] = row
        print(f'  thr={thr:.6f} ({candidates[thr]:>10})  '
              f"P={res['precision']:.4f}  R={res['recall']:.4f}  F={res['fscore']:.4f}")
        if best is None or row[args.metric] > best[1][args.metric]:
            best = (thr, row)

    best_thr, best_row = best

    # ------------------------------------------------------------- outputs
    os.makedirs(args.log, exist_ok=True)

    percentile_report['candidates'] = {f'{k:.8f}': v for k, v in candidates.items()}
    with open(os.path.join(args.log, 'percentiles.json'), 'w') as fp:
        json.dump(percentile_report, fp, indent=2)

    with open(os.path.join(args.log, 'results_by_threshold.json'), 'w') as fp:
        json.dump({
            'tuning_split': args.split,
            'eval_group': args.eval_group,
            'min_distance': args.min_distance,
            'max_distance': args.max_distance,
            'metric': args.metric,
            'best_threshold_abs': best_thr,
            'best': best_row,
            'results': results_by_threshold,
        }, fp, indent=2)

    params = {
        'mode': 'abs',
        'min_distance': args.min_distance,
        'threshold_abs': float(best_thr),
        'threshold_rel': None,
    }
    with open(os.path.join(args.log, 'params.yaml'), 'w') as fp:
        yaml.dump(params, fp)

    print('-' * 60)
    print(f'BEST THRESHOLD (threshold_abs) = {best_thr:.8f}')
    print(f'  source     : {best_row["source"]}')
    print(f'  min_distance: {args.min_distance}')
    print(f'  {args.metric} on {args.eval_group}: {best_row[args.metric]:.4f} '
          f'(P={best_row["precision"]:.4f} R={best_row["recall"]:.4f} F={best_row["fscore"]:.4f})')
    print(f'  written     : {os.path.join(args.log, "params.yaml")}')
    print('-' * 60)
    print('finish')
    print('-' * 60)


if __name__ == '__main__':
    main()
