import argparse
import os
import imageio
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='path to dataset')
parser.add_argument('output', help='output path for .h5 file')
parser.add_argument('--train', default='train.txt')
parser.add_argument('--val', default='val.txt')
parser.add_argument('--test', default='test.txt')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--sigma', type=float, default=3, help='Gaussian kernel size in pixels')
parser.add_argument('--csv_dir', default=None,
                    help='folder with the GT csv files, used for all splits (default: <dataset>/csv)')
parser.add_argument('--test_csv_dir', default=None,
                    help='folder with the GT csv files for the test split only (default: same as --csv_dir)')
parser.add_argument('--bands', default='6ch', choices=['6ch', 'RGB', 'RGBN'],
                    help='bands stored in the hdf5, selected from the 6 input channels DB,B,G,R,RE,NIR (default: 6ch)')
args = parser.parse_args()

SOURCE_BANDS = 'DB,B,G,R,RE,NIR'
# indices in the 6-channel source image; RGB/RGBN are stored in R,G,B(,N) order
BAND_INDICES = {
    '6ch': [0, 1, 2, 3, 4, 5],
    'RGB': [3, 2, 1],
    'RGBN': [3, 2, 1, 5],
}


_test_prefix_cache = {}


def uses_test_prefix(csv_dir):
    if csv_dir not in _test_prefix_cache:
        _test_prefix_cache[csv_dir] = any(
            f.startswith('test_') and f.endswith('.csv') for f in os.listdir(csv_dir))
    return _test_prefix_cache[csv_dir]


def find_csv(dataset_path, name, csv_dir):
    if csv_dir is None:
        return os.path.join(dataset_path, 'csv', name + '.csv')

    # custom folder: csv may be named without the 'test_' prefix used in test.txt.
    # The fallback is used only for folders without any 'test_*.csv', otherwise a test
    # image without trees would pick up the csv of a train image with the same base name.
    candidates = [name]
    if name.startswith('test_') and not uses_test_prefix(csv_dir):
        candidates.append(name[len('test_'):])
    for candidate in candidates:
        path = os.path.join(csv_dir, candidate + '.csv')
        if os.path.exists(path):
            return path
    return os.path.join(csv_dir, name + '.csv')


def load_data(dataset_path, names, sigma, csv_dir=None):
    data = []
    n_csv = 0
    n_points = 0

    pbar = tqdm.tqdm(total=len(names))
    for name in names:
        image = None

        for suffix in ['.tif', '.tiff', '.png']:
            image_path = os.path.join(dataset_path, 'images', name + suffix)
            if os.path.exists(image_path):
                image = imageio.imread(image_path)
                break

        if image is None:
            raise RuntimeError(f'could not find image for {name}')

        # CHECK RIGIDO: devono essere 6 bande
        if image.ndim != 3 or image.shape[-1] != 6:
            raise RuntimeError(
                f'{name}: expected image with 6 channels, got shape {image.shape}'
            )

        image = image[..., BAND_INDICES[args.bands]]

        csv_path = find_csv(dataset_path, name, csv_dir)

        if os.path.exists(csv_path):
            points = np.loadtxt(csv_path, delimiter=',', skiprows=1).astype('int')

            if len(points.shape) == 1:
                points = points[None, :]

            n_csv += 1
            n_points += len(points)

            gt = np.zeros(image.shape[:2], dtype='float32')
            gt[points[:, 1], points[:, 0]] = 1

            distance = distance_transform_edt(1 - gt).astype('float32')
            confidence = np.exp(-distance**2 / (2 * sigma**2))
        else:
            gt = np.zeros(image.shape[:2], dtype='float32')
            confidence = np.zeros(image.shape[:2], dtype='float32')

        confidence = confidence[..., None]

        attention = (confidence > 0.001).astype('float32')

        data.append({
            'name': name,
            'image': image,
            'gt': gt,
            'confidence': confidence,
            'attention': attention
        })

        pbar.update(1)

    pbar.close()
    print(f'{len(names)} images, {n_csv} with csv, {n_points} trees (csv: {csv_dir or os.path.join(dataset_path, "csv")})')
    if len(names) > 0 and n_csv == 0:
        raise RuntimeError(f'no csv found in {csv_dir or os.path.join(dataset_path, "csv")} for this split: wrong folder or names?')

    return data


def augment_images(images):
    """Rotate + flip (works with any number of channels)"""
    augmented = np.concatenate((
        images,
        np.rot90(images, k=1, axes=(1, 2)),
        np.rot90(images, k=2, axes=(1, 2)),
        np.rot90(images, k=3, axes=(1, 2))
    ))

    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented


def read_names(filename):
    path = os.path.join(args.dataset, filename)
    return [name.rstrip() for name in open(path, 'r')]


train_names = read_names(args.train)
val_names   = read_names(args.val)
test_names  = read_names(args.test)

test_csv_dir = args.test_csv_dir if args.test_csv_dir is not None else args.csv_dir

selected_bands = [SOURCE_BANDS.split(',')[i] for i in BAND_INDICES[args.bands]]
print(f'dataset : {args.dataset}')
print(f'output  : {args.output}')
print(f'bands   : {args.bands} -> source channels {BAND_INDICES[args.bands]} = {",".join(selected_bands)} (source: {SOURCE_BANDS})')
print(f'sigma   : {args.sigma}   augment: {args.augment}')

print(f'--- train ({args.train}) ---')
train_data = load_data(args.dataset, train_names, args.sigma, args.csv_dir)
print(f'--- val ({args.val}) ---')
val_data   = load_data(args.dataset, val_names, args.sigma, args.csv_dir)
print(f'--- test ({args.test}) ---')
test_data  = load_data(args.dataset, test_names, args.sigma, test_csv_dir)


def add_data_to_h5(f, data, split, augment=False):
    if len(data) == 0:
        return

    names = np.array([d['name'] for d in data])
    images = np.stack([d['image'] for d in data], axis=0)
    gt = np.stack([d['gt'] for d in data], axis=0)
    confidence = np.stack([d['confidence'] for d in data], axis=0)
    attention = np.stack([d['attention'] for d in data], axis=0)

    if augment:
        names = np.repeat(names, 8)
        images = augment_images(images)
        gt = augment_images(gt)
        confidence = augment_images(confidence)
        attention = augment_images(attention)

    f.create_dataset(f'{split}/names', data=names.astype('S'))
    f.create_dataset(f'{split}/images', data=images)
    f.create_dataset(f'{split}/gt', data=gt)
    f.create_dataset(f'{split}/confidence', data=confidence)
    f.create_dataset(f'{split}/attention', data=attention)


with h5py.File(args.output, 'w') as f:
    add_data_to_h5(f, train_data, 'train', augment=args.augment)
    add_data_to_h5(f, val_data, 'val')
    add_data_to_h5(f, test_data, 'test')

    # Salva metadata esplicito (utile davvero)
    if args.bands == '6ch':
        f.attrs['bands'] = SOURCE_BANDS
    else:
        f.attrs['bands'] = args.bands
        f.attrs['source_bands'] = SOURCE_BANDS
    f.attrs['num_channels'] = len(BAND_INDICES[args.bands])

    print(f'saved {args.output}')
    print('attrs:', dict(f.attrs))
    for split in f:
        print(f'  {split}: images {f[split]["images"].shape}, gt pixels = {int(f[split]["gt"][:].sum())}')
