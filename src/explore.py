from train_all import *

from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

import seaborn as sns
import matplotlib.pyplot as plt


def run_exploration(path_data_: str, path_wav_: str, path_results_: str, filter_: dict, feat_conf_: dict, seed_: int):
    # Check and create the directory to save the experiment
    exp_name = f'exploration_{feat_conf_["feature_type"]}_plus_{feat_conf_["extra_features"]}_' \
               f'{filter_["audio_type"][0].replace(r"/", "")}_{filter_["audio_moment"][0]}'
    path_results_ = os.path.join(path_results_, exp_name)
    os.makedirs(path_results_, exist_ok=True)
    print(f"Making the exploration of the data:"
          f" *Feature type: {feat_conf_['feature_type']}"
          f" *Extra feats: {feat_conf_['extra_features']}"
          f" *Audio type: {filter_['audio_type'][0]}"
          f" *Audio moment: {filter_['audio_moment'][0]}")

    # Define the data to be used
    dicoperia_metadata = pd.read_csv(path_data_, decimal=',')
    # Make the metadata for the DICOPERIA dataset
    path_exp_metadata = os.path.join(path_results_, 'exp_metadata.csv')
    exp_metadata = make_dicoperia_metadata(path_exp_metadata, dicoperia_metadata, filter_)
    # Make the subsets
    sample_gain_testing = 0.2
    train, test, label_train, label_test = make_train_test_subsets(exp_metadata, sample_gain_testing, seed_)

    # Make the features
    train_feats, train_labels = make_feats(path_wav_, train, label_train, feat_conf_)

    # Run the decompositions methods
    run_decomposition(train_feats, train_labels, path_results_)

    # Run the feature selection
    models = [LogisticRegression(C=0.01,
                                 max_iter=40,
                                 solver="liblinear",
                                 penalty="l2",
                                 class_weight="balanced",
                                 random_state=seed_),
              SVC(kernel="linear"),
              SVC(kernel="rbf"),
              RandomForestClassifier(n_estimators=20,
                                     criterion="gini",
                                     max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     max_features="sqrt",
                                     class_weight="balanced",
                                     random_state=seed_)
              ]
    param_grids = [{
        # Logistic Regression
        "features__pca__n_components": list(range(1, train_feats.shape[1], 5)),
        "features__univ_select__k": list(range(1, train_feats.shape[1], 5)),
        f"{models[0].__str__()}__C": [0.1, 1, 10],
        f"{models[0].__str__()}__penalty": ["l1", "l2"],
        f"{models[0].__str__()}__max_iter": [40, 80, 120]},
        # SVM Linear
        {"features__pca__n_components": list(range(1, train_feats.shape[1], 5)),
         "features__univ_select__k": list(range(1, train_feats.shape[1], 5)),
         f"{models[0].__str__()}__C": [0.1, 1, 10]},
        # SVM RBF
        {"features__pca__n_components": list(range(1, train_feats.shape[1], 5)),
         "features__univ_select__k": list(range(1, train_feats.shape[1], 5)),
         f"{models[0].__str__()}__C": [0.1, 1, 10],
         f"{models[0].__str__()}__gamma": [0.1, 1, 10]},
        # Random Forest
        {"features__pca__n_components": list(range(1, train_feats.shape[1], 5)),
         "features__univ_select__k": list(range(1, train_feats.shape[1], 5)),
         f"{models[0].__str__()}__n_estimators": [10, 20, 30],
         f"{models[0].__str__()}__max_depth": [None, 5, 10, 15],
         f"{models[0].__str__()}__min_samples_split": [2, 5, 10],
         f"{models[0].__str__()}__min_samples_leaf": [1, 2, 4],
         f"{models[0].__str__()}__max_features": ["sqrt", "log2", None]}
    ]

    for model, param_grid in zip(models, param_grids):
        best_config = run_feats_selection(train_feats, train_labels, model_=model, param_grid_=param_grid)
        print(f"Best configuration: {best_config}")


def run_decomposition(feats: pd.DataFrame, labels: np.ndarray, path_save: str = 'exploration_results'):
    """
    Run the decomposition methods to explore the data
    """

    def run_pca(data_frame, feat_columns, path_save_, pca_method):
        pca_result = pca_method.fit_transform(data_frame[feat_columns].values)
        data_frame['pca-one'] = pca_result[:, 0]
        data_frame['pca-two'] = pca_result[:, 1]
        data_frame['pca-three'] = pca_result[:, 2]
        # Plot the PCA
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x='pca-one', y='pca-two', hue="labels", data=data_frame, legend='full', alpha=0.3,
                        palette=sns.color_palette('hls', len(data_frame['labels'].unique())))
        plt.savefig(os.path.join(path_save_, f'{pca_method.__str__()}.png'))
        plt.close()
        # Plot the 3D PCA
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_frame['pca-one'], data_frame['pca-two'], data_frame['pca-three'],
                   c=data_frame['labels'], cmap='tab10')
        ax.view_init(30, 185)
        plt.savefig(os.path.join(path_save_, f'{pca_method.__str__()}_3D.png'))
        plt.close()

    def run_tsne(data_frame, feat_columns, path_save_):
        tsne = TSNE(n_components=3, verbose=1, perplexity=100, n_iter=300)
        tsne_results = tsne.fit_transform(data_frame[feat_columns].values)
        data_frame['tsne-one'] = tsne_results[:, 0]
        data_frame['tsne-two'] = tsne_results[:, 1]
        data_frame['tsne-three'] = tsne_results[:, 2]
        # Plot the TSNE
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x='tsne-one', y='tsne-two', hue="labels", data=data_frame, legend='full', alpha=0.3,
                        palette=sns.color_palette('hls', len(data_frame['labels'].unique())))
        plt.savefig(os.path.join(path_save_, 'tsne.png'))
        plt.close()
        # Plot the 3D TSNE
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_frame['tsne-one'], data_frame['tsne-two'], data_frame['tsne-three'],
                   c=data_frame['labels'], cmap='tab10')
        ax.view_init(30, 185)
        plt.savefig(os.path.join(path_save_, 'tsne_3D.png'))
        plt.close()

    feat_cols = [f'coefficient_{i}' for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=feat_cols)
    df['labels'] = labels

    # Run PCA if the number of features is greater than 10
    if feats.shape[1] > 50:
        pca_methods = [PCA(n_components=3),
                       KernelPCA(n_components=3, kernel='linear'),
                       SparsePCA(n_components=3),
                       TruncatedSVD(n_components=3, n_iter=10)]

        for pca in pca_methods:
            run_pca(df, feat_cols, path_save, pca)

    # Run the TSNE
    run_tsne(df, feat_cols, path_save)


def run_feats_selection(feats: pd.DataFrame, labels: pd.DataFrame, model_=None, param_grid_: dict = None):
    # Do PCA over the original representation:
    max_componentes: int = feats.shape[1]
    pca = PCA(n_components=max_componentes)

    # Select the k-best feats from the original representation
    k_best: int = max_componentes // 2
    selection = SelectKBest(k=k_best)

    # Build estimator from PCA and Uni-variate selection:
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Use combined features to transform dataset:
    new_features = combined_features.fit(feats, labels).transform(feats)
    print("Combined space has", new_features.shape[1], "features")

    # Define a model
    if model_ is None:
        model_ = SVC(kernel="linear")
        model_name = 'svm'
    else:
        model_name = model_.__str__()

    # Do grid search over k, n_components and C:
    pipeline = Pipeline([("features", combined_features), (model_name, model_)])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid_, verbose=10)
    grid_search.fit(feats, labels)
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_


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
    all_feats = ['ComParE_2016_voicing', 'ComParE_2016_energy', 'ComParE_2016_basic_spectral', 'ComParE_2016_spectral',
                 'ComParE_2016_mfcc', 'ComParE_2016_rasta',
                 'MFCC', 'MelSpec', 'logMelSpec']
    extra_features = [True, False]
    seed = feature_config['seed']

    # Run the exploration
    for feat in all_feats:
        if 'ComParE_2016' in feat:
            extra_features = [False]
            feature_config['compute_deltas'] = False
            feature_config['compute_deltas_deltas'] = False

        for extra in extra_features:
            feature_config['feature_type'] = feat
            feature_config['extra_features'] = extra
            for f in all_filters:
                print('================================================================\n'
                      f'Running experiment with:\n'
                      f'    Seed    : {seed}\n'
                      f'    Filters : {f}\n'
                      f'    Features: {feat} extra:{extra}\n'
                      f'----------------------------------------------------------------\n')
                run_exploration(metadata_path, wav_path, results_path, f, feature_config, seed)
