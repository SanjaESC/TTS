from multiprocessing import Pool, cpu_count
import tqdm
import argparse

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.datasets.preprocess import load_wav_data


def filter_short_samples(wav_file):
    temp_list = []
    audio = ap.load_wav(wav_file)
    mel = ap.melspectrogram(audio)
    if mel.shape[1] > 64:
        temp_list.append(wav_file)
    return temp_list


if __name__ == "__main__":
    # This script will filter out short training samples and write the training data to a file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        help='Path to config file with link to training data.',
                        required=True)
    args = parser.parse_args()
    c = load_config(args.config_path)
    temp_eval_data, temp_train_data = load_wav_data(c.data_path, c.eval_split_size)
    full_training_data = temp_eval_data + temp_train_data
    # setup audio processor
    ap = AudioProcessor(**c.audio)

    with Pool(cpu_count()-1) as p:
        # filter train_data

        train_data = list(tqdm.tqdm(p.imap(filter_short_samples, full_training_data), total=len(full_training_data)))
        train_data = [i for sub_list in train_data for i in sub_list]
        print(f' - > Filtered {len(full_training_data) - len(train_data)} instances.')

    with open('training_set.csv', 'a') as w_file:
        for data in train_data:
            w_file.write(data+'\n')