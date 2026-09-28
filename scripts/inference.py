from models import SFANet
from utils.preprocess import preprocess_RGB, preprocess_RGBN
from utils.preprocess_6ch import preprocess_6bands
from utils.inference import run_tiled_inference

import argparse
import os
import yaml

import tqdm

import glob

# bands -> (preprocess function, number of channels read from the raster)
BANDS = {
    'RGB': (preprocess_RGB, 3),
    'RGBN': (preprocess_RGBN, 4),
    '6ch': (preprocess_6bands, 6),
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='path to input tiff file or directory')
    parser.add_argument('output', help='path to output json file or directory')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--bands', default='RGBN', choices=list(BANDS),
                        help='input bands: RGB, RGBN or 6ch (raster bands ordered as in training; 6ch = DB,B,G,R,RE,NIR)')
    parser.add_argument('--tile_size', type=int, default=2048, help='tile size')
    parser.add_argument('--overlap', type=int, default=32, help='overlap between tiles')

    args = parser.parse_args()

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
        min_distance = 1
        threshold_abs = None
        threshold_rel = 0.2
    
    weights_path = SFANet.find_weights(args.log)
    padded_size = args.tile_size + args.overlap*2
    preprocess, nbands = BANDS[args.bands]
    training_model, model = SFANet.build_model((padded_size,padded_size,nbands),preprocess_fn=preprocess)
    training_model.load_weights(weights_path)

    tiling = dict(tile_size=args.tile_size,overlap=args.overlap)
    if os.path.isdir(args.input):
        os.makedirs(args.output,exist_ok=True)
        paths = sorted(glob.glob(os.path.join(args.input,'*.tif')) + glob.glob(os.path.join(args.input,'*.tiff')))
        pbar = tqdm.tqdm(total=len(paths))
        for input_path in paths:
            output_path = os.path.join(args.output,os.path.basename(input_path).split('.')[0]+'.json')
            if not os.path.exists(output_path):
                run_tiled_inference(model,input_path,output_path,min_distance=min_distance,threshold_abs=threshold_abs,threshold_rel=threshold_rel,**tiling)
            pbar.update(1)
    else:
        run_tiled_inference(model,args.input,args.output,min_distance=min_distance,threshold_abs=threshold_abs,threshold_rel=threshold_rel,**tiling)

if __name__ == '__main__':
    main()
