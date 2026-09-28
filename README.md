## Urban Tree Detection (ResNet50 + SFANet, TensorFlow) ##

Tree detection from multispectral aerial imagery: a SFANet-style network with a ResNet50 encoder outputs a confidence map, and local peak finding gives the tree locations.

### Workflow ###

    prepare_6ch  ->  train  ->  tune_percentile  ->  calculate_ap  ->  test

| Step | Script | Output (in `<log>` unless noted) |
|------|--------|-----------------------------------|
| 1 | `scripts/prepare_6ch.py` | `<data>` (hdf5 file) |
| 2 | `scripts/train.py` | `best.weights.h5`, `latest.weights.h5` |
| 3 | `scripts/tune_percentile.py` | `params.yaml`, `percentiles.json`, `results_by_threshold.json` |
| 4 | `scripts/calculate_ap.py` | `ap_results.txt`, `precision_recall_curve.png` |
| 5 | `scripts/test.py` | `results.txt` |

Run all commands from the root of the repository. `<data>` is the hdf5 file, `<log>` the log directory.

Weights are saved as `best.weights.h5` / `latest.weights.h5` (Keras 3 naming); log directories trained with the previous version (`weights.best.h5`) are still read by all scripts.

### Installation ###

Requires TensorFlow >= 2.16 (Keras 3); tested with Python 3.13, TensorFlow 2.21.0, Keras 3.15.0.

With conda:

    conda env create
    conda activate urban-tree-detection

With a Python virtual environment (venv):

    python3.13 -m venv ~/.virtualenvs/urbantree
    source ~/.virtualenvs/urbantree/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

If the GPU is not found (`Cannot dlopen some GPU libraries` in the log), export the CUDA libraries installed by pip (after activating the environment; needed in every new shell):

    # conda
    export LD_LIBRARY_PATH=$(ls -d $CONDA_PREFIX/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
    # venv
    export LD_LIBRARY_PATH=$(ls -d $VIRTUAL_ENV/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH

Check that TensorFlow sees the GPUs:

    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

GPU selection: `train.py` uses GPU 0, `calculate_ap.py` and `test.py` GPU 1 (set in the code), `tune_percentile.py` has `--gpu` (default 1).

### 1. Prepare the dataset ###

    python3 -m scripts.prepare_6ch <dataset> <data> --bands <6ch|RGB|RGBN> --sigma 6 --augment --csv_dir <csv folder>

Dataset layout:

    <dataset>/images/<name>.tif        6-channel images, bands DB,B,G,R,RE,NIR
    <dataset>/<csv folder>/<name>.csv  x,y tree locations, one header line (no csv = no trees)
    <dataset>/train.txt, val.txt, test.txt   one image name per line

| Option | Meaning |
|--------|---------|
| `--bands` | `6ch` (default): all bands; `RGB`: R,G,B; `RGBN`: R,G,B,NIR. RGB and RGBN are picked from the 6 channels |
| `--augment` | 4 rotations and a flip (x8), training split only |
| `--sigma` | Gaussian size (pixels) of the confidence maps; pass it explicitly |
| `--csv_dir` | folder with the ground-truth csv, all splits (default `<dataset>/csv`) |
| `--test_csv_dir` | folder with the ground-truth csv for the test split only (e.g. all trees instead of a filtered ground truth) |
| `--train`, `--val`, `--test` | split files (defaults `train.txt`, `val.txt`, `test.txt`); an empty file gives an hdf5 without that split |

The script stops if no csv is found for a split. Test csv files without the `test_` prefix used in `test.txt` are matched automatically. `tune_percentile.py` needs the `val` split, `calculate_ap.py` and `test.py` the `test` split.

### 2. Training ###

    python3 -m scripts.train <data> <log> [--lr 1e-4] [--epochs 500] [--batch_size 8]

### 3. Percentile tuning ###

Chooses the absolute detection threshold on the **validation** set:

1. predicts the confidence maps of the validation images;
2. splits them into images with and without trees (from the ground truth);
3. computes the percentiles (default 90, 92, 95, 97, 99) of the values clipped to [0, 1] for each group: these are the candidate thresholds;
4. runs detection and matching for every candidate and keeps the best F-score.

<!-- -->

    python3 -m scripts.tune_percentile <data> <log> [--cache_preds]

Options: `--percentiles`, `--min_distance 3`, `--max_distance 20` (pixels), `--metric fscore|precision|recall`, `--eval_group all|tree`, `--split val`, `--gpu`, `--weights`. `--cache_preds` saves the predictions in `<log>/val_preds.npy` and reuses them.

The result is written to `<log>/params.yaml` (`mode: abs`, `min_distance`, `threshold_abs`, `threshold_rel`), read by `calculate_ap.py`, `test.py` and `inference.py`.

### 4. Average precision ###

    python3 -m scripts.calculate_ap <data> <log> [--max_distance 20]

Average precision on the test set over a range of thresholds. The point of the precision-recall curve closest to the `threshold_abs` in `<log>/params.yaml` is marked as `TUNED` (and reported in `ap_results.txt`); without `params.yaml` only the best-F1 point is shown.

### 5. Test ###

    python3 -m scripts.test <data> <log> [--max_distance 20] [--dataset <dataset>]

Applies the parameters in `<log>/params.yaml` to the test set and writes precision, recall, F-score and RMSE (pixels) to `<log>/results.txt`.

With `--dataset` (the folder with `images/<name>.tif` used by `prepare_6ch`), it also saves the geo-referenced outputs of every test image in `<log>`: the confidence map in `Saliency_Map/Sal_Map_<name>.tif` and the true positives, false positives and false negatives in `Labeled_prediction/<name>_labaled_pred.geojson`.

### Inference on a large raster ###

    python3 -m scripts.inference <input tiff or directory> <output json or directory> <log> --bands <RGB|RGBN|6ch>

Produces GeoJSON files with the geo-referenced trees, using the best weights and `params.yaml` from `<log>`. The raster bands must be in the same order used in training: `6ch` expects `DB,B,G,R,RE,NIR`, `RGB` and `RGBN` expect R,G,B(,NIR) as first bands.

### License ###

This repository is released under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) (see `LICENSE`): it can be used, modified and shared for noncommercial purposes only.

It is derived from [jonathanventura/urban-tree-detection](https://github.com/jonathanventura/urban-tree-detection), Copyright (c) 2022 Jonathan Ventura, released under the MIT License (see `LICENSE-MIT`). The portions of the code coming from that repository remain available under the MIT License.

### Citation ###

This work builds upon the urban tree detection method and code by Ventura et al. If you use this repository, please also cite their paper:

J. Ventura, C. Pawlak, M. Honsberger, C. Gonsalves, J. Rice, N.L.R. Love, S. Han, V. Nguyen, K. Sugano, J. Doremus, G.A. Fricker, J. Yost, and M. Ritter (2024). [Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery.](https://www.sciencedirect.com/science/article/pii/S1569843224002024) International Journal of Applied Earth Observation and Geoinformation, 130, 103848.
