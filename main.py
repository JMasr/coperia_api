import os.path

from apscheduler.schedulers.blocking import BlockingScheduler

from src.util import *
from src.data import MyPatient, CoperiaMetadata


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
    audios_metadata = CoperiaMetadata(audios_dataset[0]).metadata

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
    coperia_audios = []
    for sample_rate in [48000, 16000]:
        coperia_audio = process_coperia_audio(patients=patients,
                                              sample_rate=sample_rate,
                                              audio_observations=observations,
                                              path_save=os.path.join(root_path, f'wav_{sample_rate}kHz'))

        data_path = os.path.join(root_path, f'coperia_audios_{sample_rate}.pkl')
        save_obj(data_path, coperia_audio)
        coperia_audios.append(coperia_audio)
    return coperia_audios


def download_coperia_patients(root_path: str, observations: list) -> dict:
    """
    Download and store in a dictionary the patient's metadata given a list of observation
    :param observations: list with the observation
    :param root_path: root path of the data directory
    :return: a dictionary where the key is the patient's id and the value is an instance of the class MyPatient
    """
    path = os.path.join(root_path, f'patients.pkl')

    patients_dict = {}
    for observation in observations:
        patient_id = observation.subject.reference.split('/')[-1]
        if patient_id not in patients_dict.keys():
            patient = MyPatient(observation)
            patients_dict[patient_id] = patient
    save_obj(path, patients_dict)
    return patients_dict


def download_coperia_observations(root_path: str) -> list:
    """
    Download the observation of Coperia by the voice codes, save it as pickle files, and return it as a list
    :param root_path: root path of the data directory
    :return: a list with two elements, each with the observation of one Coperia voice code
    """
    dataset = []
    api = CoperiaApi(os.getcwd())

    for code in ['84435-7', '84728-5']:
        data = api.get_observations_by_code(code)
        dataset.extend(data)

        print(f"+=== Downloaded {len(data)} observations with code {code}. ===+")
        path_audios = os.path.join(root_path, f'audio_obs_{code}.pkl')
        save_obj(path_audios, data)
    return dataset


def check_4_new_data(path_data: str, codes: list = None):
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


def update_data(root_path: str = 'dataset_V4'):
    """
    Check for new data in the Coperia Cloud and update the local files of the dataset
    :param root_path: root path of the data directory
    """

    if check_4_new_data(root_path):
        print("There are new data.")
        observations = download_coperia_observations(root_path)
        patients = download_coperia_patients(root_path, observations)
        audios_dataset = make_audios_dataset(root_path, observations, patients)
        audios_metadata = make_audios_metadata(root_path, audios_dataset)
        make_metadata_plots(audios_metadata, root_path)
        make_audios_spectrogram(audios_metadata, root_path)
        print("Dataset update!")
        return True
    else:
        print("There isn't new data.")
        return False


if __name__ == "__main__":
    # scheduler = BlockingScheduler()
    # scheduler.add_job(update_data('dataset_V4'), 'interval', hours=24)
    # scheduler.start()

    update_data('dataset_V4')
