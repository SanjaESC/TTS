import os
import time
import torch
import json
from datetime import datetime, date
import string
from glob import glob
import numpy as np
from pathlib import Path
import sys
import random
import argparse


from TTS.utils.synthesis import synthesis
from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config, load_checkpoint
from TTS.utils.text.symbols import make_symbols, symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.text.text_cleaning import clean_sentence
from TTS.vocoder.utils.generic_utils import setup_generator 


def tts(model,
        vocoder_model,
        C,
        VC,
        text,
        ap,
        ap_vocoder,
        use_cuda,
        batched_vocoder,
        speaker_id=None,
        style_input=None,
        figures=False):
    use_vocoder_model = vocoder_model is not None

    waveform, alignment, _, postnet_output, stop_tokens, _ = synthesis(
        model, text, C, use_cuda, ap, speaker_id, style_input=style_input,
        truncated=False, enable_eos_bos_chars=C.enable_eos_bos_chars,
        use_griffin_lim=(not use_vocoder_model), do_trim_silence=True)


    if C.model == "Tacotron" and use_vocoder_model:
        postnet_output = ap.out_linear_to_mel(postnet_output.T).T
    # correct if there is a scale difference b/w two models
    
    if use_vocoder_model:
        vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
        waveform = vocoder_model.inference(vocoder_input)
        if use_cuda:
            waveform = waveform.cpu()
        #waveform = waveform.detach().numpy()
        waveform = waveform.numpy()
        waveform = waveform.flatten()
        

    # if use_vocoder_model:
    #     postnet_output = ap._denormalize(postnet_output)
    #     postnet_output = ap_vocoder._normalize(postnet_output)
    #     vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
    #     waveform = vocoder_model.generate(
    #         vocoder_input.cuda() if use_cuda else vocoder_input,
    #         batched=batched_vocoder,
    #         target=8000,
    #         overlap=400)

    return alignment, postnet_output, stop_tokens, waveform


def load_vocoder(lib_path, model_file, model_config, use_cuda):
    sys.path.append(lib_path) # set this if ParallelWaveGAN is not installed globally
    #pylint: disable=import-outside-toplevel
    vocoder_config = load_config(model_config)
    vocoder_model = setup_generator(vocoder_config)
    checkpoint = torch.load(model_file, map_location='cpu')
    print(' > Model step:', checkpoint['step'])
    vocoder_model.load_state_dict(checkpoint['model'])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0
    vocoder_config = load_config(model_config)
    ap_vocoder = AudioProcessor(**vocoder_config['audio'])

    if use_cuda:
        vocoder_model.cuda()
    return vocoder_model.eval(), ap_vocoder


def split_into_sentences(text):
    text = text.replace('.', '.<stop>')
    text = text.replace('!', '!<stop>')
    text = text.replace('?', '?<stop>')
    sentences = text.split("<stop>")
    sentences = list(filter(None, [s.strip() for s in sentences]))  # remove empty sentences
    return sentences


def main(**kwargs):
    global symbols, phonemes # pylint: disable=global-statement
    current_date = date.today()
    current_date = current_date.strftime("%B %d %Y")
    start_time = time.time()

    # read passed variables from gui
    text = kwargs['text']                           # text to generate speech from
    use_cuda = kwargs['use_cuda']                   # if gpu exists default is true
    project = kwargs['project']                     # path to project folder
    vocoder_type = kwargs['vocoder']                # vocoder type, default is GL
    use_gst = kwargs['use_gst']                     # use style_wave for prosody
    style_dict = kwargs['style_input']              # use style_wave for prosody
    speaker_id = kwargs['speaker_id']               # name of the selected speaker
    sentence_file = kwargs['sentence_file']         # path to file if generate from file
    out_path = kwargs['out_path']                   # path to save the output wav

    batched_vocoder = True

    # load speakers
    speakers_file_path = Path(project, "speakers.json")
    if speakers_file_path.is_file():
        speaker_data = json.load(open(speakers_file_path, 'r'))
        num_speakers = len(speaker_data)
        #get the speaker id for selected speaker
        if speaker_id >= num_speakers:
            print('Speaker ID outside of number of speakers range. Using default 0.')
            speaker_id = 0
            speaker_name = [speaker for speaker, id in speaker_data.items() if speaker_id == id][0]
        else:
            speaker_name = [speaker for speaker, id in speaker_data.items() if speaker_id == id][0]
    else:
        speaker_name = 'Default'
        num_speakers = 0
        speaker_id = None

    # load the config
    config_path = Path(project, "config.json")
    C = load_config(config_path)

    if use_gst:
        if style_dict is not None:
            style_input = style_dict
    else:
        style_input = None

    # load the audio processor
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    if 'characters' in C.keys():
        symbols, phonemes = make_symbols(**C.characters)
        

    # find the tts model file in project folder
    try:
        tts_model_file = glob(str(Path(project, '*.pth.tar')))
        if not tts_model_file:
            raise FileNotFoundError
        model_path = tts_model_file[0]
    except FileNotFoundError:
        print('[!] TTS Model not found in path: "{}"'.format(project))

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C)

    # if gpu is not available use cpu
    model, state = load_checkpoint(model, model_path, use_cuda=use_cuda)

    model.decoder.max_decoder_steps = 2000

    model.eval()
    print(' > Model step:', state['step'])
    print(' > Model r: ', state['r'])

    # load vocoder
    if vocoder_type is 'MelGAN':
        try:
            model_file = glob(str(Path(project, 'vocoder/*.pth.tar')))
            vocoder, ap_vocoder = load_vocoder(str(Path('TTS')),
                                               str(model_file[0]),
                                               str(Path(project, 'vocoder/config.json')),
                                               use_cuda)
        except Exception:
            print('[!] Error loading vocoder: "{}"'.format(project))
            sys.exit(0)

    elif vocoder_type is 'WaveRNN':
        try:
            model_file = glob(str(Path(project, 'vocoder/*.pkl')))
            vocoder, ap_vocoder = load_vocoder(str(Path('TTS')), str(model_file[0]), str(Path(project, 'config.yml')), use_cuda)
        except Exception:
            print('[!] Error loading vocoder: "{}"'.format(project))
            sys.exit(0)
    else:
        vocoder, ap_vocoder = None, None

    print(" > Vocoder: {}".format(vocoder_type))
    print(' > Using style input: {}\n'.format(style_input))

    if sentence_file != '':
        with open(sentence_file, "r", encoding='utf8') as f:
            list_of_sentences = [s.strip() for s in f.readlines()]
    else:
        list_of_sentences = [text.strip()]

    # iterate over every passed sentence and synthesize
    for _, tts_sentence in enumerate(list_of_sentences):
        wav_list = []
        # remove character which are not alphanumerical or contain ',. '
        tts_sentence = clean_sentence(tts_sentence) 
        print(" > Text: {}".format(tts_sentence))
        # build filename
        current_time = datetime.now().strftime("%H%M%S")
        file_name = ' '.join(tts_sentence.split(" ")[:10])
        # if multiple sentences in one line -> split them
        tts_sentence = split_into_sentences(tts_sentence)
        
        # if sentence was split in sub-sentences -> iterate over them
        for sentence in tts_sentence:
            # synthesize voice
            _, _, _, wav = tts(model,
                               vocoder,
                               C,
                               None,
                               sentence,
                               ap,
                               ap_vocoder,
                               use_cuda,
                               batched_vocoder,
                               speaker_id=speaker_id,
                               style_input=style_input,
                               figures=False)

            # join sub-sentences back together and add a filler between them
            wav_list += list(wav)
            wav_list += [0] * 10000

        wav = np.array(wav_list)

        # finalize filename
        file_name = "_".join([str(current_time), file_name])
        file_name = file_name.translate(
            str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'

        if out_path == "":
            out_dir = str(Path(project, 'output', current_date, speaker_name))
            out_path = os.path.join(out_dir, file_name)
        else:
            out_dir = os.path.dirname(out_path)

        # create output directory if it doesn't exist
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # save generated wav to disk
        ap.save_wav(wav, out_path)
        end_time = time.time()
        print(" > Run-time: {}".format(end_time - start_time))
        print(" > Saving output to {}\n".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',
                        type=str,
                        required=True,
                        help='Text to generate speech.')
    parser.add_argument('--project_path',
                        type=str,
                        required=True,
                        help='Path to model')
    parser.add_argument('--use_gst',
                        type=bool,
                        help='Use Global Style Tokens.',
                        default=False)
    parser.add_argument('--use_cuda',
                        type=bool,
                        help='Run model on CUDA.',
                        default=False)
    parser.add_argument('--style_input',
                        type=bool,
                        help='Use style input.',
                        default=False)
    parser.add_argument('--vocoder_type',
                        type=str,
                        help='Vocoder Type -> Choices: [Griffin-Lim, WaveRNN, MelGAN].',
                        default="MelGAN")
    parser.add_argument('--speaker_id',
                        type=int,
                        help='Select speaker by ID.',
                        default=0)
    parser.add_argument('--sentence_file',
                        type=str,
                        help='Path to text file to generate speech from.',
                        default="")
    parser.add_argument('--out_path',
                        type=str,
                        help='Path to save final wav file.',
                        default="")
    args = parser.parse_args()

    main(text=args.text,
         use_cuda=args.use_cuda,
         use_gst=args.use_gst,
         style_input=args.style_input,
         project=args.project_path,
         speaker_id=args.speaker_id,
         vocoder=args.vocoder_type,
         sentence_file=args.sentence_file,
         out_path=args.out_path)
