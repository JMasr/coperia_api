import argparse
import datetime
import os.path

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from src.train import FeatureExtractor
from src.util import load_config_from_json


def mlflow_grid_search(feats, labels, all_models, exp_pipeline, exp_metrics, num_folds, data_name, time):
    print("Run the grid search for each model")
    for model_name, model_config in all_models.items():
        # Set the model in the pipeline
        model = model_config['model']
        exp_pipeline.set_params(model=model)

        # Set up the grid search
        num_cpus = os.cpu_count()//4
        print(f'Start GRID-SEARCH on model: ', {model_name}, ' with ', {num_cpus}, ' CPUs')
        grid_search = GridSearchCV(exp_pipeline, param_grid=model_config['params'], cv=num_folds, scoring=exp_metrics,
                                   refit='roc_auc', verbose=1, n_jobs=num_cpus)
        print("Fit data")
        with mlflow.start_run(nested=True, run_name=f'{model_name}-GridSearch'):
            # Automatically log all relevant information about the model
            mlflow.sklearn.autolog()
            grid_search.fit(feats, labels)

            # Log the results in MLflow
            params = {'model': model_name, 'dataset': data_name, 'date': time, **grid_search.best_params_}
            mlflow.set_tags(params)
            mlflow.log_params(params)

            # Log the mean and std scores for each metric
            for metric in exp_metrics:
                metric_name = 'mean_' + metric + '_score'
                mlflow.log_metric(metric_name, grid_search.cv_results_['mean_test_' + metric][grid_search.best_index_])
                metric_name = 'std_' + metric + '_score'
                mlflow.log_metric(metric_name, grid_search.cv_results_['std_test_' + metric][grid_search.best_index_])


def make_feats(df_data: pd.DataFrame, feats_configuration: dict) -> (np.ndarray, np.ndarray):
    """
    Given a dataframe with columns 'audio_id' and 'patient_type', make for each audio file the features and the labels
    :param df_data: Dataframe with the data *MUST HAVE THE COLUMNS 'audio_id' AND 'patient_type'*
    :param feats_configuration: Configuration of the features to be extracted
    :return: Features and labels
    """
    # Feature extractor
    features_extractor = FeatureExtractor(feats_configuration)

    # Extract features
    egs = []
    for row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        audio_path = row[1]['audio_id']
        label = row[1]['patient_type']

        # Prepare features
        x_feats = np.array(features_extractor.extract(audio_path))
        x_labels = np.array([label] * x_feats.shape[0]).reshape(x_feats.shape[0], 1)

        # Make the examples
        egs.append(np.concatenate((x_feats, x_labels), axis=1))
    # Prepare the data
    egs = np.vstack(egs)
    return np.array(egs[:, :-1], dtype=float), np.array(egs[:, -1], dtype=int)


def make_subsets(path_csv: str, path_audio: str, splitting_factor: float = 0.2, random_state: int = 42,
                 shuffle: bool = True) -> (pd.DataFrame, pd.DataFrame):
    """
    Make the train and test subsets from the data
    :param path_csv: Path to the csv file
    :param path_audio: Path to the audio files
    :param splitting_factor: Factor to split the data
    :param random_state: Random state
    :param shuffle: Shuffle the data
    :return: Train and test data
    """
    # Read the data
    df = pd.read_csv(path_csv, decimal=',')
    df.replace(['covid-control', 'covid-persistente'], [0, 1], inplace=True)
    # Add to the columns the path to the wav file
    df['audio_id'] = df['audio_id'].apply(lambda wav_id: os.path.join(path_audio, wav_id + '.wav'))
    # Split the data
    if splitting_factor > 0:
        patient_and_labels = df[['patient_id', 'patient_type']].drop_duplicates()
        pat_train, pat_test, _, _ = train_test_split(patient_and_labels['patient_id'],
                                                     patient_and_labels['patient_type'],
                                                     test_size=splitting_factor,
                                                     stratify=patient_and_labels['patient_type'],
                                                     random_state=random_state,
                                                     shuffle=shuffle)
    else:
        pat_train = df['patient_id'].unique()
        pat_test = []
    # Get the train and test data
    train_set = df[df['patient_id'].isin(pat_train)][['audio_id', 'patient_type']]
    test_set = df[df['patient_id'].isin(pat_test)][['audio_id', 'patient_type']]
    return train_set, test_set


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='/home/jsanhcez/Documentos/Proyectos/06_TODO_COPERIA/repos/coperia_api')
    args = parser.parse_args()

    # Define the data to be used
    path_root = args.r
    path_data = os.path.join(path_root, 'dataset_dicoperia/dicoperia_metadata.csv')
    path_wav = wav_path = os.path.join(path_root, 'dataset_dicoperia/wav_48000kHz/')

    print("Define training parameters")
    seed: int = load_config_from_json(os.path.join(path_root, 'config', 'run_config.json'))['seed']
    k_folds: int = load_config_from_json(os.path.join(path_root, 'config', 'run_config.json'))['k_folds']
    test_size: float = load_config_from_json(os.path.join(path_root, 'config', 'run_config.json'))['test_size']
    # Define the features to be used
    feats_config = load_config_from_json(os.path.join(path_root, 'config', 'feature_config.json'))

    print("Make the train and test data")
    path_feats_np = os.path.join(path_root, 'dataset_dicoperia/dicoperia_all-feats_and_labels.npy')
    if os.path.exists(path_feats_np):
        # Load the data
        df_feats = np.load(path_feats_np, allow_pickle=True)
        x, y = df_feats[:, :-1], df_feats[:, -1]
    else:
        train_data, test_data = make_subsets(path_data, path_wav, splitting_factor=test_size, random_state=seed)
        # Make the features
        x, y = make_feats(train_data, feats_config)
        # Save the data
        np.save(path_feats_np, np.concatenate((x, y.reshape(y.shape[0], 1)), axis=1))

    print("Define the models to be tested")
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {
                'model__C': [0.01, 0.1, 1.0, 10, 100],
                'model__max_iter': [1000],
                'model__penalty': ['l2', 'none'],
                'model__solver': ['lbfgs', 'sag', 'saga'],
                'model__class_weight': [None, 'balanced'],
                'kbest__k': [100, 75, 50],
                'kbest__score_func': [f_classif, mutual_info_classif],
                'PCA__n_components': [100, 75, 50]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'model__n_estimators': [100, 500, 1000],
                'model__criterion': ['gini', 'entropy'],
                'model__max_depth': [5, 10, 20, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['auto', 'sqrt'],
                'model__class_weight': [None, 'balanced'],
                'kbest__k': [5, 10, 15],
                'kbest__score_func': [f_classif, mutual_info_classif],
                'PCA__n_components': [5, 10, 15]
            }
        },
        'LinearSVM': {
            'model': SVC(),
            'params': {
                'model__C': [0.1, 1.0, 10.0],
                'model__tol': [1e-3, 1e-4, 1e-5],
                'model__max_iter': [100, 500, 1000],
                'model__class_weight': [None, 'balanced'],
                'kbest__k': [5, 10, 15],
                'kbest__score_func': [f_classif, mutual_info_classif],
                'PCA__n_components': [5, 10, 15]
            }
        },
        'RBF-SVM': {
            'model': SVC(kernel='rbf'),
            'params': {
                'model__C': [0.1, 1.0, 10.0],
                'model__tol': [1e-3, 1e-4, 1e-5],
                'model__gamma': [0.01, 0.1, 1.0, 'scale'],
                'model__max_iter': [100, 500, 1000],
                'model__class_weight': [None, 'balanced'],
                'kbest__k': [5, 10, 15],
                'kbest__score_func': [f_classif, mutual_info_classif],
                'PCA__n_components': [5, 10, 15]
            }
        },
        'MLP': {
            'model': MLPClassifier(),
            'params': {
                'model__hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
                'model__activation': ['relu', 'tanh', 'logistic'],
                'model__solver': ['adam', 'lbfgs'],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__max_iter': [1000],
                'kbest__k': [5, 10, 15],
                'kbest__score_func': [f_classif, mutual_info_classif],
                'PCA__n_components': [5, 10, 15]
            }

        }
    }

    # Define the pipeline
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('kbest', SelectKBest(score_func=f_classif, k=10)),
                         ('PCA', PCA(n_components=10)),
                         ('model', None)
                         ])

    # Define the evaluation metrics
    metrics = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc']

    # Set up the mlflow experiment
    mlflow.set_tracking_uri('http://localhost:5000')
    experiment_name = 'covid_classification_test'
    mlflow.set_experiment(experiment_name)
    # Define some metadata for the experiment
    dataset_name = 'DICOPERIA-ALL'
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # Run the exp
    mlflow_grid_search(x, y, models, pipeline, metrics, k_folds, dataset_name, date)
