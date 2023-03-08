import json
import os
import pickle
import random
import string

import pandas
import torch
import torchaudio
import opensmile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, f1_score, confusion_matrix, \
    precision_recall_curve, ConfusionMatrixDisplay, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm import tqdm
from sklearn.model_selection import train_test_split


class FeatureExtractor:
    """
    Class for feature extraction
    args: input arguments dictionary
    Mandatory arguments: resampling_rate, feature_type, window_size, hop_length
    For MFCC: f_max, n_mels, n_mfcc
    For MelSpec/logMelSpec: f_max, n_mels
    Optional arguments: compute_deltas, compute_delta_deltas
    """

    def __init__(self, args: dict):

        self.args = args
        self.audio_path = None
        self.resampling_rate = self.args['resampling_rate']
        assert (args['feature_type'] in ['MFCC', 'MelSpec', 'logMelSpec',
                                         'ComParE_2016_llds', 'ComParE_2016_voicing', 'ComParE_2016_spectral',
                                         'ComParE_2016_mfcc', 'ComParE_2016_rasta', 'ComParE_2016_basic_spectral',
                                         'ComParE_2016_energy']), \
            'Expected the feature_type to be MFCC / MelSpec / logMelSpec / ComParE_2016'

        nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
        hop_length = int(float(self.args.get('hop_length', 0)) * 1e-3 * self.resampling_rate)

        if self.args['feature_type'] == 'MFCC':
            self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate,
                                                                n_mfcc=int(self.args['n_mfcc']),
                                                                melkwargs={'n_fft': nfft,
                                                                           'n_mels': int(self.args['n_mels']),
                                                                           'f_max': int(self.args['f_max']),
                                                                           'hop_length': hop_length})
        elif self.args['feature_type'] in ['MelSpec', 'logMelSpec']:
            self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resampling_rate,
                                                                          n_fft=nfft,
                                                                          n_mels=int(self.args['n_mels']),
                                                                          f_max=int(self.args['f_max']),
                                                                          hop_length=hop_length)
        elif 'ComParE_2016' in self.args['feature_type']:
            self.feature_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                                     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                                     sampling_rate=self.resampling_rate)
        else:
            raise ValueError('Feature type not implemented')

    def _read_audio(self, filepath):
        """ This code does the following:
                1. Read audio,
                2. Resample the audio if required,
                3. Perform waveform normalization,
                4. Compute sound activity using threshold based method
                5. Discard the silence regions
        :param filepath: path to the audio file
        :return: a torch.Tensor with the audio samples and an int with the sample rate
        """

        s, fs = torchaudio.load(filepath)
        if fs != self.resampling_rate:
            s, fs = torchaudio.sox_effects.apply_effects_tensor(s, fs, [['rate', str(self.resampling_rate)]])
        if s.shape[0] > 1:
            s = s.mean(dim=0).unsqueeze(0)
        s = s / torch.max(torch.abs(s))
        sad = self.compute_sad(s.numpy(), self.resampling_rate)
        s = s[np.where(sad == 1)]
        return s, fs

    @staticmethod
    def compute_sad(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
        """ Compute threshold based sound activity """
        # Leading/Trailing margin
        sad_start_end_sil_length = int(sad_start_end_sil_length * 1e-3 * fs)
        # Margin around active samples
        sad_margin_length = int(sad_margin_length * 1e-3 * fs)

        sample_activity = np.zeros(sig.shape)
        sample_activity[np.power(sig, 2) > threshold] = 1
        sad = np.zeros(sig.shape)
        for i in range(sample_activity.shape[1]):
            if sample_activity[0, i] == 1:
                sad[0, i - sad_margin_length:i + sad_margin_length] = 1
        sad[0, 0:sad_start_end_sil_length] = 0
        sad[0, -sad_start_end_sil_length:] = 0
        return sad

    def _do_feature_extraction(self, s, fs):
        """ Feature preparation
        Steps:
        1. Apply feature extraction to waveform
        2. Convert amplitude to dB if required
        3. Append delta and delta-delta features
        """
        F = None

        if 'ComParE_2016' in self.args['feature_type']:
            #
            s = s[None, :]
            # get a random string
            file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            while os.path.exists(file_name):
                file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            torchaudio.save(file_name + '.wav', s, sample_rate=self.resampling_rate)
            F = self.feature_transform.process_file(file_name + '.wav')

            # columns based selection
            os.remove(file_name + '.wav')

            # feature subsets
            feature_subset = {}
            if self.args['feature_type'] == 'ComParE_2016_voicing':
                feature_subset['subset'] = ['F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                                            'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma']

            if self.args['feature_type'] == 'ComParE_2016_energy':
                feature_subset['subset'] = ['audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
                                            'pcm_RMSenergy_sma', 'pcm_zcr_sma']

            if self.args['feature_type'] == 'ComParE_2016_spectral':
                feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
                                            'audSpec_Rfilt_sma[3]', 'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]',
                                            'audSpec_Rfilt_sma[6]', 'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]',
                                            'audSpec_Rfilt_sma[9]', 'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]',
                                            'audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]', 'audSpec_Rfilt_sma[14]',
                                            'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]','audSpec_Rfilt_sma[17]',
                                            'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
                                            'audSpec_Rfilt_sma[21]', 'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]',
                                            'audSpec_Rfilt_sma[24]','audSpec_Rfilt_sma[25]',
                                            'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                                            'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
                                            'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
                                            'pcm_fftMag_spectralFlux_sma',
                                            'pcm_fftMag_spectralCentroid_sma',
                                            'pcm_fftMag_spectralEntropy_sma',
                                            'pcm_fftMag_spectralVariance_sma',
                                            'pcm_fftMag_spectralSkewness_sma',
                                            'pcm_fftMag_spectralKurtosis_sma',
                                            'pcm_fftMag_spectralSlope_sma',
                                            'pcm_fftMag_psySharpness_sma',
                                            'pcm_fftMag_spectralHarmonicity_sma',
                                            'mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
                                            'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]', 'mfcc_sma[9]', 'mfcc_sma[10]',
                                            'mfcc_sma[11]', 'mfcc_sma[12]','mfcc_sma[13]', 'mfcc_sma[14]']

            if self.args['feature_type'] == 'ComParE_2016_mfcc':
                feature_subset['subset'] = ['mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
                                            'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]',
                                            'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]',
                                            'mfcc_sma[13]', 'mfcc_sma[14]']

            if self.args['feature_type'] == 'ComParE_2016_rasta':
                feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
                                            'audSpec_Rfilt_sma[3]',
                                            'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]', 'audSpec_Rfilt_sma[6]',
                                            'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]', 'audSpec_Rfilt_sma[9]',
                                            'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]', 'audSpec_Rfilt_sma[12]',
                                            'audSpec_Rfilt_sma[13]',
                                            'audSpec_Rfilt_sma[14]', 'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]',
                                            'audSpec_Rfilt_sma[17]',
                                            'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
                                            'audSpec_Rfilt_sma[21]',
                                            'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]', 'audSpec_Rfilt_sma[24]',
                                            'audSpec_Rfilt_sma[25]']

            if self.args['feature_type'] == 'ComParE_2016_basic_spectral':
                feature_subset['subset'] = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                                            'pcm_fftMag_spectralRollOff25.0_sma',
                                            'pcm_fftMag_spectralRollOff50.0_sma',
                                            'pcm_fftMag_spectralRollOff75.0_sma',
                                            'pcm_fftMag_spectralRollOff90.0_sma',
                                            'pcm_fftMag_spectralFlux_sma',
                                            'pcm_fftMag_spectralCentroid_sma',
                                            'pcm_fftMag_spectralEntropy_sma',
                                            'pcm_fftMag_spectralVariance_sma',
                                            'pcm_fftMag_spectralSkewness_sma',
                                            'pcm_fftMag_spectralKurtosis_sma',
                                            'pcm_fftMag_spectralSlope_sma',
                                            'pcm_fftMag_psySharpness_sma',
                                            'pcm_fftMag_spectralHarmonicity_sma']

            if self.args['feature_type'] == 'ComParE_2016_llds':
                feature_subset['subset'] = list(F.columns)

            F = F[feature_subset['subset']].to_numpy()
            F = np.nan_to_num(F)
            F = torch.from_numpy(F).T

        if self.args['feature_type'] == 'MelSpec':
            F = self.feature_transform(s)

        if self.args['feature_type'] == 'logMelSpec':
            F = self.feature_transform(s, fs)
            F = torchaudio.functional.amplitude_to_DB(F, multiplier=10, amin=1e-10, db_multiplier=0)

        if self.args['feature_type'] == 'MFCC':
            F = self.feature_transform(s)

        if self.args.get('compute_deltas', False):
            FD = torchaudio.functional.compute_deltas(F)
            F = torch.cat((F, FD), dim=0)

            if self.args.get('compute_delta_deltas', False):
                FDD = torchaudio.functional.compute_deltas(FD)
                F = torch.cat((F, FDD), dim=0)

        if feature_config.get('apply_mean_norm', False):
            F = F - torch.mean(F, dim=0)

        if feature_config.get('apply_var_norm', False):
            F = F / torch.std(F, dim=0)

        # own feature selection
        if self.args.get('extra_feats', False):

            # Make a temporary file to save the audio to
            file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            while os.path.exists(file_name):
                file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            s = s[None, :]
            torchaudio.save(file_name + '.wav', s, sample_rate=self.resampling_rate)
            # Config OpenSMILE
            feature_subset = {'subset': [
                # Voicing
                'F0final_sma', 'voicingFinalUnclipped_sma',
                'jitterLocal_sma', 'jitterDDP_sma',
                'shimmerLocal_sma',
                'logHNR_sma',
                # Energy
                'audspec_lengthL1norm_sma',
                'audspecRasta_lengthL1norm_sma',
                'pcm_RMSenergy_sma',
                'pcm_zcr_sma',
                # Spectral
                'pcm_fftMag_fband250-650_sma',
                'pcm_fftMag_fband1000-4000_sma',
                'pcm_fftMag_spectralRollOff25.0_sma',
                'pcm_fftMag_spectralRollOff50.0_sma',
                'pcm_fftMag_spectralRollOff75.0_sma',
                'pcm_fftMag_spectralRollOff90.0_sma',
                'pcm_fftMag_spectralFlux_sma',
                'pcm_fftMag_spectralCentroid_sma',
                'pcm_fftMag_spectralEntropy_sma',
                'pcm_fftMag_spectralVariance_sma',
                'pcm_fftMag_spectralSkewness_sma',
                'pcm_fftMag_spectralKurtosis_sma',
                'pcm_fftMag_spectralSlope_sma',
                'pcm_fftMag_psySharpness_sma',
                'pcm_fftMag_spectralHarmonicity_sma'
            ]}
            extra_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                              sampling_rate=self.resampling_rate)
            # Extract features
            F_extra = extra_transform.process_file(file_name + '.wav')
            F_extra = F_extra[feature_subset['subset']].to_numpy()
            F_extra = np.nan_to_num(F_extra)
            F_extra = torch.from_numpy(F_extra).T
            # Concatenate the features
            common_shape = min(F.shape[1], F_extra.shape[1])
            F = torch.cat((F[:, :common_shape], F_extra[:, :common_shape]), dim=0)
            # Remove the temporary file
            os.remove(file_name + '.wav')

        return F.T

    def extract(self, filepath):
        """
        Extracts the features from the audio file
        :param filepath: path to the audio file
        :return: features
        """
        self.audio_path = filepath
        s, fs = self._read_audio(filepath)
        return self._do_feature_extraction(s, fs)


def run_exp(path_data_: str, path_wav_: str, path_results_: str, filters: dict, feature_config_: dict, model_name: str,
            models: dict, random_state: int = 42):
    # Check and create the directory to save the experiment
    os.makedirs(path_results_, exist_ok=True)

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

    # Train the model
    path_model = os.path.join(path_results_, f'{model_name}_results')
    if not os.path.exists(os.path.join(path_model, f'{model_name}.pkl')):
        # Create the directory to save the results
        os.makedirs(path_model, exist_ok=True)
        # Configure the model
        model, x_train, y_train = config_model(model_name, models, train_feats, train_labels)
        # Train the model
        model.fit(x_train, y_train)
        # Save the model
        pickle.dump(model, open(os.path.join(path_model, f'{model_name}.pkl'), 'wb'))
    else:
        model = pickle.load(open(os.path.join(path_model, f'{model_name}.pkl'), 'rb'))

    # Calculate the performance metrics using sklearn
    all_scores = score_sklearn(model, [test, label_test], path_wav_,
                               os.path.join(path_model, f'{model_name}_scores.pkl'))
    return all_scores


def score_sklearn(model_, test_data: list, path_wav: str, path_to_save: str) -> dict:
    """
    Calculate the performance metrics using sklearn
    :param model_: a model trained using for inference
    :param test_data: a list of test data and labels
    :param path_wav: Path to the wav files
    :param path_to_save: Path to save the results
    :return: a set of performance metrics: confusion matrix, f1 score, f-beta score, precision, recall, and auc score
    """
    model_name = model_.__class__.__name__
    # Start testing
    y_score, y_true = [], []
    for ind, audio_id in tqdm(test_data[0].items(), total=test_data[0].shape[0]):
        # Prepare features
        FE = FeatureExtractor(feature_config)
        F = FE.extract(os.path.join(path_wav, audio_id + '.wav'))

        # Predict
        if exp_model == 'LSTMclassifier':
            with torch.no_grad():
                feat = F.to('cpu')
                output_score = model_.predict_proba(feat)
                output_score = sum(output_score)[0].item() / len(output_score)
        else:
            output_score = model_.predict(F)
            output_score = float(np.mean(output_score))

        # Average the scores of all segments from the input file
        y_score.append(output_score)
        y_true.append(test_data[1][ind])

    # Calculate the auc_score, FP-rate, and TP-rate
    sklearn_roc_auc_score = roc_auc_score(y_true, y_score)
    sklear_fpr, sklearn_tpr, n_thresholds = roc_curve(y_true, y_score)

    # calculate the specificity and sensitivity
    sensitivity = sklearn_tpr[1]
    specificity = 1 - sensitivity

    # Make prediction using a threshold that maximizes the difference between TPR and FPR
    optimal_idx = np.argmax(sklearn_tpr - sklear_fpr)
    optimal_threshold = n_thresholds[optimal_idx]
    y_pred = [1 if scr > optimal_threshold else 0 for scr in y_score]

    # Calculate Precision, Recall, F1, and F-beta scores
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred)
    f1_scr = f1_score(y_true, y_pred)

    # Calculate Confusion Matrix
    confusion_mx = confusion_matrix(y_true, y_pred)

    dict_scores = {'model_name': model_name,
                   'acc_score': float(acc),
                   'tpr': sklearn_tpr.tolist(),
                   'tnr': sklear_fpr.tolist(),
                   'sensitivity': float(sensitivity),
                   'specificity': float(specificity),
                   'decision_thresholds': n_thresholds.tolist(),
                   'optimal_threshold': float(optimal_threshold),
                   'auc_score': float(sklearn_roc_auc_score),
                   'confusion_matrix': confusion_mx.tolist(),
                   'f1_scr': float(f1_scr),
                   'f_beta': f_beta.tolist(),
                   'precision': precision.tolist(),
                   'recall': recall.tolist()}

    # Save the results
    if path_to_save is not None:
        # Save dict_scores
        with open(path_to_save, "wb") as f:
            pickle.dump(dict_scores, f)
        # Save dict_scores as a human-readable txt file
        with open(path_to_save.replace('.pkl', '.json'), 'w') as f:
            pretty_score = json.dumps(dict_scores, indent=4)
            f.write(pretty_score)

    # Plot useful metric graphs
    if path_to_save is not None:
        # Plot the ROC curve for the model
        plt.plot(sklear_fpr, sklearn_tpr, marker='.', label=model_name)
        plt.title('ROC Curve')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # Save the ROC curve plot
        plt.savefig(path_to_save.replace('.pkl', '_ROC.png'))
        plt.close()

        # Plot the Precision-Recall curve for the model
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(lr_precision, lr_recall, label=model_name)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the title
        plt.title(f'Precision-Recall Curve of {model_name} (Threshold={optimal_threshold:.2f}')
        plt.savefig(path_to_save.replace('.pkl', '_precision_recall.png'))
        plt.close()

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mx, display_labels=['Control', 'Positive'])
        disp.plot()
        # Plot the Confusion Matrix for the threshold selected
        plt.title(f'CM of {model_name} (Threshold={optimal_threshold:.2f}')
        plt.savefig(path_to_save.replace('.pkl', '_confusion_matrix.png'))
        plt.close()

    print("--------------------------------------------")
    print("Scoring: Accuracy = {:.2f}, AUC = {:.2f}".format(acc, sklearn_roc_auc_score))
    print('============================================\n')
    return dict_scores


def config_model(model_name: str, models: dict, training_feats, training_labels):
    """
    Function to configure a model
    @param model_name: Type of the model
    @param models: Dictionary with all the available models
    @param training_feats: Training features
    @param training_labels: Training labels
    @return: Configured model
    """
    model_args = models[model_name]

    if model_name == 'LogisticRegression':
        model = LogisticRegression(C=float(model_args['c']),
                                   max_iter=int(model_args['max_iter']),
                                   solver=model_args['solver'],
                                   penalty=model_args['penalty'],
                                   class_weight=model_args['class_weight'],
                                   random_state=model_args['random_state'],
                                   verbose=True)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=model_args['n_estimators'],
                                       criterion=model_args['criterion'],
                                       max_depth=model_args['max_depth'],
                                       min_samples_split=model_args['min_samples_split'],
                                       min_samples_leaf=model_args['min_samples_leaf'],
                                       max_features=model_args['max_features'],
                                       class_weight=model_args['class_weight'],
                                       random_state=model_args['random_state'])
    elif model_name == 'LinearSVM':
        model = SVC(C=model_args['c'],
                    tol=model_args['tol'],
                    max_iter=model_args['max_iter'],
                    verbose=model_args['verbose'],
                    class_weight=model_args['class_weight'],
                    random_state=model_args['random_state'])
        #
        # trans = StandardScaler()
        # training_feats = trans.fit_transform(training_feats)

    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=model_args['hidden_layer_sizes'],
                              solver=model_args['solver'], alpha=model_args['alpha'],
                              learning_rate_init=model_args['learning_rate_init'],
                              verbose=model_args['verbose'], activation=model_args['activation'],
                              max_iter=model_args['max_iter'], random_state=model_args['random_state'])

        if model_args['class_weight'] == 'balanced':
            train_data = np.concatenate((training_feats, training_labels.reshape(training_feats.shape[0], 1)),
                                        axis=1)
            ind = np.where(train_data[:, -1] == 1)[0]
            n_positives = len(ind)
            n_negatives = train_data.shape[0] - n_positives
            up_sample_factor = int(n_negatives / n_positives) - 1
            for i in range(up_sample_factor):
                train_data = np.concatenate((train_data, train_data[ind, :]), axis=0)
            np.random.shuffle(train_data)
            training_feats = train_data[:, :-1]
            training_labels = train_data[:, -1]
    else:
        raise ValueError("Not implementation of the model: " + model_name)
    return model, training_feats, training_labels


def make_feats(path_to_wav: str, audio_id: pd.DataFrame, labels: pd.DataFrame, feats_config: dict):
    """
    Extract features from audio files
    :param path_to_wav: path to audio files
    :param audio_id: audio ids as a pandas dataframe
    :param labels: labels as a pandas dataframe
    :param feats_config: feature configuration as a dictionary
    """
    # Prepare feature extractor
    FE = FeatureExtractor(feats_config)

    egs = []

    data = pandas.concat([audio_id, labels], axis=1)
    for row in tqdm(data.iterrows(), total=data.shape[0]):
        audio_id = row[1]['audio_id']
        label = row[1]['patient_type']
        # Prepare features
        F = FE.extract(os.path.join(path_to_wav, audio_id + '.wav'))
        egs.append(np.concatenate((np.array(F), np.array([label] * F.shape[0]).reshape(F.shape[0], 1)), axis=1))
    egs = np.vstack(egs)

    return np.array(egs[:, :-1], dtype=float), np.array(egs[:, -1], dtype=int)


def make_train_test_subsets(metadata: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Making the subsets by patients
    PATIENT_ID_COLUMN = 'patient_id'
    CLASS_COLUMN = 'patient_type'
    AUDIO_ID_COLUMN = 'audio_id'

    # Prepare the metadata
    patient_data = metadata[[PATIENT_ID_COLUMN, CLASS_COLUMN]].drop_duplicates()
    patient_id = patient_data[PATIENT_ID_COLUMN]
    patient_class = patient_data[CLASS_COLUMN]

    # Split the data
    pat_train, pat_test, pat_labels_train, pat_labels_test = train_test_split(patient_id, patient_class,
                                                                              test_size=test_size,
                                                                              random_state=random_state,
                                                                              stratify=patient_class)
    # Using the patient subsets to select the audio samples
    audio_data_train = metadata[(metadata[PATIENT_ID_COLUMN].isin(pat_train))]
    audio_train = audio_data_train[AUDIO_ID_COLUMN]
    audio_label_train = audio_data_train[CLASS_COLUMN]

    audio_data_test = metadata[(metadata[PATIENT_ID_COLUMN].isin(pat_test))]
    audio_test = audio_data_test[AUDIO_ID_COLUMN]
    audio_label_test = audio_data_test[CLASS_COLUMN]

    # Print the final length of each subset
    print(f"Test-set: {len(pat_test)} patients & {len(audio_data_test)} samples")
    print(f"Train-set: {len(pat_train):} patients & {len(audio_data_train)} samples")
    return audio_train, audio_test, audio_label_train, audio_label_test


def make_dicoperia_metadata(save_path: str, metadata: pd.DataFrame, filters_: dict = None, remove_samples: dict = None):
    """
    Make a metadata file for the COPERIA dataset filtering some columns
    :param save_path: path to save the metadata file
    :param metadata: a list with all the audio samples in COPERIA as an Audio class
    :param filters_: a dictionary with the columns and values to keep
    :param remove_samples: a dictionary with the columns and values to remove
    :return: a pandas dataframe with the metadata of the DICOPERIA dataset
    """
    print('=== Filtering the metadata... ===')
    df = metadata.copy()

    if filters_ is None:
        filters_ = {'patient_type': ['covid-control', 'covid-persistente']}

    if remove_samples is None:
        remove_samples = {'audio_id': ['c15e54fc-5290-4652-a3f7-ff3b779bd980', '244b61cc-4fd7-4073-b0d8-7bacd42f6202'],
                          'patient_id': ['coperia-rehab']}

    for key, values in remove_samples.items():
        df = df[~df[key].isin(values)]

    for key, values in filters_.items():
        df = df[df[key].isin(values)]

    df.replace(['covid-control', 'covid-persistente'], [0, 1], inplace=True)
    df.to_csv(save_path, index=False, decimal=',')
    print('Metadata saved in: {}'.format(save_path))
    print('=== Filtering DONE!! ===\n')
    return df


def load_config_from_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config_as_json(config: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Define important paths
    root_path = '/home/jsanhcez/Documentos/Proyectos/99_to_do_COPERIA/repos/coperia_api/'
    data_path = os.path.join(root_path, 'dataset_dicoperia/')
    wav_path = os.path.join(data_path, 'wav_48000kHz/')
    metadata_path = os.path.join(data_path, 'metadata_dicoperia.csv')
    # Data filters
    all_filters = load_config_from_json(os.path.join(root_path, 'config', 'filter_config.json'))
    # Feature configuration
    feature_config = load_config_from_json(os.path.join(root_path, 'config', 'feature_config.json'))

    # Models configurations
    seed = int(abs(hash(str(feature_config))) / 1e12)
    all_models = load_config_from_json(os.path.join(root_path, 'config', 'models_config.json'))

    # Run the experiments
    for exp_filter in all_filters:
        print('================================' + '=' * len(str(exp_filter)))
        print(f'Running experiment with filter: {exp_filter}')
        print('--------------------------------' + '-' * len(str(exp_filter)))

        results_path = os.path.join(root_path, f'results_{feature_config["feature_type"]}_{exp_filter["audio_type"][0].replace(r"/","")}_{exp_filter["audio_moment"][0]}_{seed}/')
        feature_config['output_path'] = results_path
        for m in all_models.keys():
            print('================================' + '=' * len(m))
            print(f'Running experiment with model: {m}')
            print('--------------------------------' + '-' * len(m))
            run_exp(metadata_path, wav_path, results_path, exp_filter, feature_config, m, all_models, seed)
