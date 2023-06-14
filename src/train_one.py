import argparse
import os

from src.util import load_config_from_json
from src.train_all import run_exp

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='/home/jsanhcez/Documentos/Proyectos/06_TODO_COPERIA/repos/coperia_api')
    args = parser.parse_args()

    # Define important paths
    root_path = args.r
    data_path = os.path.join(root_path, 'dataset/')
    csv_path = os.path.join(data_path, 'dicoperia_metadata.csv')
    wav_path = os.path.join(data_path, 'wav_48000kHz/')
    results_path = os.path.join(root_path, 'results')

    # Define model parameters
    model_name = 'RandomForest'
    k_folds, seed = 0, 42

    # Define filters
    filters = {"audio_type": ["/cough/"], "audio_moment": ["before"]}

    # Define features
    feats_config = load_config_from_json(os.path.join(root_path, 'config', 'feature_config.json'))
    feats_config['feature_type'] = 'MFCC'
    feats_config['extra_features'] = False

    run_exp(csv_path, wav_path, results_path, filters, feats_config, model_name, k_folds, seed)
