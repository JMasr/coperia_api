import argparse
import os

from src.api import update_data
from src.train_all import run_exp
from src.util import load_config_from_json

if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()
    # Define task
    parser.add_argument('--donwload', '-d', type=bool, default=False)
    parser.add_argument('--train', '-t', type=bool, default=True)
    # Define paths
    parser.add_argument('-r', default=os.getcwd())
    parser.add_argument('--data_path', '-o', type=str, default='dataset')
    args = parser.parse_args()

    if args.donwload:
        # Check for new data
        update_data(args.data_path)
    elif args.train:
        # Train the model
        root_path = args.r
        data_path = os.path.join(root_path, 'dataset/')
        wav_path = os.path.join(data_path, 'wav_48000kHz/')
        csv_path = os.path.join(data_path, 'dicoperia_metadata.csv')
        results_path = os.path.join(root_path, 'results')
        # Data filters
        all_filters = load_config_from_json(os.path.join(root_path, 'config', 'filter_config.json'))
        # Feature configuration
        feats_config = load_config_from_json(os.path.join(root_path, 'config', 'feature_config.json'))
        # Models configurations
        MODELS = load_config_from_json(os.path.join(root_path, 'config', 'models_config.json'))
        # Runs configuration
        run_config = load_config_from_json(os.path.join(root_path, 'config', 'run_config.json'))
        num_folds = run_config['k_folds']
        random_state = run_config['seed']

        # Run the experiments
        for exp_filter in all_filters:
            # Select the feats
            all_feats = ['MFCC', 'MelSpec', 'logMelSpec',
                         'ComParE_2016_voicing', 'ComParE_2016_energy',
                         'ComParE_2016_basic_spectral', 'ComParE_2016_spectral',
                         'ComParE_2016_mfcc', 'ComParE_2016_rasta',
                         'Spafe_mfcc', 'Spafe_imfcc', 'Spafe_cqcc', 'Spafe_gfcc', 'Spafe_lfcc',
                         'Spafe_lpc', 'Spafe_lpcc', 'Spafe_msrcc', 'Spafe_ngcc', 'Spafe_pncc',
                         'Spafe_psrcc', 'Spafe_plp', 'Spafe_rplp']
            extra_features = [True, False]

            for feat in all_feats:
                feats_config['feature_type'] = feat
                if 'ComParE_2016' in feat:
                    extra_features = [False]

                for extra in extra_features:
                    feats_config['extra_features'] = extra

                    # Select the model
                    for model_ in MODELS.keys():
                        print('================================================================'
                              f'Running experiment with: \n'
                              f'    Model   :{model_}  -  Seed:{random_state}\n'
                              f'    Filters :{exp_filter}\n'
                              f'    Features:{feat} extra:{extra}\n'
                              f'----------------------------------------------------------------')
                        run_exp(csv_path, wav_path, f'{results_path}_{random_state}',
                                exp_filter, feats_config, model_,
                                num_folds, random_state)
