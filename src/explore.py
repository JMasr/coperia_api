from train import *

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt


def run_exploration(path_data_: str, path_wav_: str, path_results_: str, filters: dict, feature_config_: dict,
                    random_state: int = 42):
    # Check and create the directory to save the experiment
    exp_name = f'exploration_{feature_config_["feature_type"]}_plus_{feature_config_["extra_feats"]}_' \
               f'{filters["audio_type"][0].replace(r"/", "")}_{filters["audio_moment"][0]}'
    path_results_ = os.path.join(path_results_, exp_name)
    os.makedirs(path_results_, exist_ok=True)
    print(f"Making the exploration of the data:"
          f" *Feature type: {feature_config_['feature_type']}"
          f" *Extra feats: {feature_config_['extra_feats']}"
          f" *Audio type: {filters['audio_type'][0]}"
          f" *Audio moment: {filters['audio_moment'][0]}")

    # Define the data to be used
    dicoperia_metadata = pd.read_csv(path_data_, decimal=',')
    # Make the metadata for the DICOPERIA dataset
    path_exp_metadata = os.path.join(path_results_, 'exp_metadata.csv')
    exp_metadata = make_dicoperia_metadata(path_exp_metadata, dicoperia_metadata, filters)
    # Make the subsets
    sample_gain_testing = 0.2
    train, test, label_train, label_test = make_train_test_subsets(exp_metadata, sample_gain_testing, random_state)

    # Make the features
    if not os.path.exists(os.path.join(path_results_, f'train_feats_{random_state}.npy')):
        train_feats, train_labels = make_feats(path_wav_, train, label_train, feature_config_)
        # Save the features
        np.save(os.path.join(path_results_, f'train_feats_{random_state}.npy'), train_feats)
        np.save(os.path.join(path_results_, f'train_labels_{random_state}.npy'), train_labels)
    else:
        train_feats = np.load(os.path.join(path_results_, f'train_feats_{random_state}.npy'))
        train_labels = np.load(os.path.join(path_results_, f'train_labels_{random_state}.npy'))

    make_pca_tsne(train_feats, train_labels, path_results_)


def make_pca_tsne(feats: pd.DataFrame, labels: np.ndarray, path_save: str = 'exploration_results'):
    """
    Make the PCA and plot it.
    """
    feat_cols = [f'coefficient_{i}' for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=feat_cols)
    df['labels'] = labels
    # Make the PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # Plot the PCA
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='pca-one', y='pca-two', hue="labels", data=df, legend='full', alpha=0.3,
                    palette=sns.color_palette('hls', len(df['labels'].unique())))
    plt.show()
    plt.savefig(os.path.join(path_save, 'pca.png'))
    plt.close()

    # Plot the 3D PCA
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['pca-one'], df['pca-two'], df['pca-three'], c=df['labels'], cmap='tab10')
    ax.view_init(30, 185)
    plt.show()
    plt.savefig(os.path.join(path_save, 'pca_3D.png'))
    plt.close()

    # Make the TSNE
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    df['tsne-three'] = tsne_results[:, 2]
    # Plot the TSNE
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='tsne-one', y='tsne-two', hue="labels", data=df, legend='full', alpha=0.3,
                    palette=sns.color_palette('hls', len(df['labels'].unique())))
    plt.show()
    plt.savefig(os.path.join(path_save, 'tsne.png'))
    plt.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/home/jsanhcez/Documentos/Proyectos/99_to_do_COPERIA/repos/coperia_api/')
    args = parser.parse_args()

    # Define important paths
    root_path = args.root_path
    data_path = os.path.join(root_path, 'dataset_dicoperia/')
    wav_path = os.path.join(data_path, 'wav_48000kHz/')
    metadata_path = os.path.join(data_path, 'metadata_dicoperia.csv')
    results_path = os.path.join(root_path, 'exploration_results/')

    # Data filters
    all_filters = load_config_from_json(os.path.join(root_path, 'config', 'filter_config.json'))
    # Feature configuration
    feature_config = load_config_from_json(os.path.join(root_path, 'config', 'feature_config.json'))

    # Run explorations
    all_feats = ['MFCC', 'MelSpec', 'logMelSpec',
                 'ComParE_2016_voicing', 'ComParE_2016_energy', 'ComParE_2016_basic_spectral', 'ComParE_2016_spectral',
                 'ComParE_2016_mfcc', 'ComParE_2016_rasta']
    extra_feats = [True, False]

    for feat in all_feats:
        for extra in extra_feats:
            feature_config['feature_type'] = feat
            feature_config['extra_features'] = extra
            for f in all_filters:
                run_exploration(metadata_path, wav_path, results_path, f, feature_config)
