from multiprocessing import Pool

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.datasets.preprocess import load_wav_data



c = load_config("/media/alexander/LinuxFS/Documents/PycharmProjects/TTS/vocoder/configs/multiband-melgan_and_rwd_config.json")

def filter_short_samples(wav_file):
    temp_list = []
    audio = ap.load_wav(wav_file)
    mel = ap.melspectrogram(audio)
    if mel.shape[1] > 64:
        temp_list.append(wav_file)
    return temp_list


temp_eval_data, temp_train_data = load_wav_data(c.data_path, c.eval_split_size)

# setup audio processor
ap = AudioProcessor(**c.audio)

with Pool(5) as p:
    # filter train_data
    train_data = p.map(filter_short_samples, temp_train_data)
    train_data = [i for sub_list in train_data for i in sub_list]
    print(f' > Filtered {len(temp_train_data) - len(train_data)} train instances.')

with Pool(5) as p:
    # filter eval_data
    eval_data = p.map(filter_short_samples, temp_eval_data)
    eval_data = [i for sub_list in eval_data for i in sub_list]
    print(f' > Filtered {len(temp_eval_data) - len(eval_data)} eval instances.')


with open('training_set.csv', 'a') as w_file:
    for data in train_data:
        w_file.write(data+'\n')
             
with open('training_set.csv', 'a') as w_file:
    for data in eval_data:
        w_file.write(data+'\n')