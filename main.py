import argparse
import os.path

from tqdm import tqdm

from src.data import CoperiaMetadata
from src.util import *


def make_inference_files(root_path: str, output_path: str, audios_metadata: pd.DataFrame):
    """
    Giving a pd.DataFrame with the audio dataset metadata, make a scp file for each group of patients
    :param root_path: root path of the data directory
    :param output_path: path where the scp files will be saved
    :param audios_metadata: a list with all the audio samples as an Audio class
    """
    print("Making scp files...")
    os.makedirs(output_path, exist_ok=True)
    # Filtering data
    patient_control = audios_metadata[audios_metadata['patient_type'] == 'covid-control']
    patient_control_auidio_type_a = patient_control[patient_control['audio_type'] == '/a/']
    patient_control_auidio_type_cough = patient_control[patient_control['audio_type'] == '/cough/']

    patient_persistente = audios_metadata[audios_metadata['patient_type'] == 'covid-persistente']
    patient_persistente_auidio_type_a = patient_persistente[patient_persistente['audio_type'] == '/a/']
    patient_persistente_auidio_type_cough = patient_persistente[patient_persistente['audio_type'] == '/cough/']

    # Making scp files
    with open(os.path.join(output_path, 'scp_all'), 'w') as f:
        for row in patient_control.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')
        for row in patient_persistente.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')

    with open(os.path.join(output_path, 'reference_all'), 'w') as f:
        for row in patient_control.itertuples():
            f.write(f'{row.audio_id}\tn\n')
        for row in patient_persistente.itertuples():
            f.write(f'{row.audio_id}\tp\n')

    with open(os.path.join(output_path, 'scp_all_a'), 'w') as f:
        for row in patient_control_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')
        for row in patient_persistente_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')

    with open(os.path.join(output_path, 'reference_all_a'), 'w') as f:
        for row in patient_control_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\tn\n')
        for row in patient_persistente_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\tp\n')

    with open(os.path.join(output_path, 'scp_all_cough'), 'w') as f:
        for row in patient_control_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')
        for row in patient_persistente_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')

    with open(os.path.join(output_path, 'reference_all_cough'), 'w') as f:
        for row in patient_control_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\tn\n')
        for row in patient_persistente_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\tp\n')


def make_audios_spectrogram(root_path: str, audios_metadata: pd.DataFrame):
    """
    Make a spectrogram of each audio in the dataset
    :param root_path: root path of the data directory
    :param audios_metadata: a list with all the audio samples as an Audio class
    """
    print('Making spectrograms...')
    path_audio = os.path.join(root_path, f'wav_48000kHz')
    path_spectrogram = os.path.join(root_path, f'wav_48000kHz_spectrogram')
    make_spectrogram(path_audio, path_spectrogram)
    struct_spectrogram(audios_metadata, path_spectrogram)


def make_metadata_plots(root_path: str, audios_metadata: pd.DataFrame):
    """
    Plot and save a set of png files with information about the dataset
    :param root_path: root path of the data directory
    :param audios_metadata: a list with all the audio samples as an Audio class
    """
    print('Making metadata...')
    coperia_metadata_control = audios_metadata[audios_metadata['patient_type'] == 'covid-control']
    coperia_metadata_persistente = audios_metadata[audios_metadata['patient_type'] == 'covid-persistente']
    print('Making plots...')
    plot_all_data([audios_metadata, coperia_metadata_control, coperia_metadata_persistente],
                  [os.path.join(root_path, 'figures_all'),
                   os.path.join(root_path, 'figures_control'),
                   os.path.join(root_path, 'figures_persistente')])


def make_audios_metadata(root_path: str, audios_dataset: list) -> pd.DataFrame:
    """
    Make a csv file with all the audio dataset metadata
    :param root_path: root path of the data directory
    :param audios_dataset: a list with all the audio samples as an Audio class
    :return: a pandas.DataFrame with the audio dataset metadata
    """
    audios_metadata = CoperiaMetadata(audios_dataset).metadata

    metadata_path = os.path.join(root_path, 'coperia_metadata')
    audios_metadata.to_csv(metadata_path, decimal=',')
    return audios_metadata


def make_audios_dataset(root_path: str, observations: list, patients: dict) -> list:
    """
    Make the audio samples and a dataset with instance of the class Audio
    :param root_path: root path of the data directory
    :param observations: list with the observation
    :param patients: a dictionary with all the patients {patient_id: MyPatient}
    :return: a list with all the audio samples as an Audio class
    """
    # Proces and save the audio data
    sample_rate = 48000
    coperia_audio = process_coperia_audio(patients=patients,
                                          sample_rate=sample_rate,
                                          audio_observations=observations,
                                          path_save=os.path.join(root_path, f'wav_{sample_rate}kHz'))

    data_path = os.path.join(root_path, f'coperia_audios_{sample_rate}.pkl')
    save_obj(data_path, coperia_audio)
    return coperia_audio


def download_coperia_patients(root_path: str, observations: list) -> dict:
    """
    Download and store in a dictionary the patient's metadata given a list of observation
    :param observations: list with the observation
    :param root_path: root path of the data directory
    :return: a dictionary where the key is the patient's id and the value is an instance of the class MyPatient
    """
    path = os.path.join(root_path, f'patients.pkl')

    patients_dict = {}
    for observation in tqdm(observations):
        patient_id = observation.subject.reference.split('/')[-1]
        if patient_id not in patients_dict.keys():
            patient = MyPatient(observation)
            patients_dict[patient_id] = patient
    save_obj(path, patients_dict)
    return patients_dict


def download_coperia_observations(root_path: str, codes: list = None) -> list:
    """
    Download the observation of Coperia by the voice codes, save it as pickle files, and return it as a list
    :param root_path: root path of the data directory
    :param codes: list with the observation codes
    :return: a list with two elements, each with the observation of one Coperia voice code
    """
    if codes is None:
        codes = ['84435-7', '84728-5']

    dataset = []
    api = CoperiaApi(os.getcwd())

    for code in codes:
        data = api.get_observations_by_code(code)
        dataset.extend(data)

        print(f"+=== Downloaded {len(data)} observations with code {code}. ===+")
        path_audios = os.path.join(root_path, f'audio_obs_{code}.pkl')
        save_obj(path_audios, data)
    return dataset


def check_4_new_data(path_data: str, codes: list = None):
    """
    Check if there is new data in the Coperia server
    :param path_data: path of the data directory
    :param codes: list with the observation codes
    """
    if codes is None:
        codes = ['84435-7', '84728-5']

    for code in codes:
        obs_path = os.path.join(path_data, f'audio_obs_{code}.pkl')
        if os.path.exists(obs_path):
            observations = load_obj(obs_path)
        else:
            observations = []

        obs_in_disk = len(observations)
        obs_in_cloud = CoperiaApi(os.getcwd()).get_observations_total(code)

        if obs_in_disk != obs_in_cloud:
            return True

    return False


def update_data(root_path: str = 'dataset_V4') -> bool:
    """
    Check for new data in the Coperia Cloud and update the local files of the dataset
    :param root_path: root path of the data directory
    """

    if check_4_new_data(root_path):
        print("There are new data.")
        observations = download_coperia_observations(root_path)
        patients = download_coperia_patients(root_path, observations)
        audio_dataset = make_audios_dataset(root_path, observations, patients)
        audio_metadata = make_audios_metadata(root_path, audio_dataset)
        make_metadata_plots(root_path, audio_metadata)
        make_audios_spectrogram(root_path, audio_metadata)
        make_inference_files(os.path.join(root_path, 'wav_48000kHz'), 'dataset/inference_files', audio_metadata)
        print("Dataset update!")
        return True
    else:
        print("There isn't new data.")
        return False


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()
    # Set a directory to save the data
    parser.add_argument('--data_path', '-o', default='dataset')
    args = parser.parse_args()
    # Check for new data
    update_data(args.data_path)
