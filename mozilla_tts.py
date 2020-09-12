import PySimpleGUI as sg
import threading
from TTS import synthesize
from torch.cuda import is_available
import json
from glob import glob
from pathlib import Path
import os
import subprocess
import platform
from datetime import date, time # pylint: disable=unused-import

# Global Variables
status = False
stop_threads = False
_platform = platform.system()
version = '0.0.2'

def synthesize_tts(text,
                   use_cuda,
                   use_gst,
                   style_input,
                   project,
                   speaker_list,
                   speakers,
                   vocoder_type,
                   sentence_file):
    global status, stop_threads # pylint: disable=global-statement
    for idx, speaker in enumerate(speaker_list):
        if speaker in speakers:
            print(speakers)
            if stop_threads:
                break
            synthesize.main(text=text,
                            use_cuda=use_cuda,
                            use_gst=use_gst,
                            style_input=style_input,
                            project=project,
                            speaker_id=idx,
                            vocoder=vocoder_type,
                            sentence_file=sentence_file,
                            out_path="")

    status, stop_threads = True, False


def open_output_folder(speaker_path):
    """[Try to determin the operating system and use the corresponding function to open a folder]

    Args:
        speaker_path ([string]): [Path to the output folder of the selected speaker]
    """
    try:
        if _platform == 'Darwin':       # macOS
            subprocess.call(('open', str(speaker_path)))  # not tested
        elif _platform == 'Windows':    # Windows
            os.startfile(str(speaker_path))
        else:   # linux variants
            subprocess.Popen(['xdg-open', str(speaker_path)])
    except FileNotFoundError as er:
        print(er)


def get_emotion_weights(selected_emotion):
    """[This function gets a selected emotion as string from the dropdown gui element.]

    Args:
        selected_emotion ([string]): [Selected emotion from the gui element, used to filter for the needed weights]

    Returns:
        [dictionary]: [Return a selected dictionary with slider-keys and the corresponding values]
    """
    # Emotion weights can be defined here
    EMOTIONS = [
        ["normal", {'speedSlider':0, 'emotionSlider': 0, 'toneSlider': 0}],
        ["angry", {'speedSlider':0, 'emotionSlider': -30, 'toneSlider': 10}],
        ["dominant", {'speedSlider':0, 'emotionSlider': 30, 'toneSlider': 10}],
        ["calm", {'speedSlider':0, 'emotionSlider': 20, 'toneSlider': -10}],
        ]
    for emotion in EMOTIONS:
        if selected_emotion in emotion[0]:
            return emotion[1]


def main_gui():
    global status, stop_threads # pylint: disable=global-statement

    # Variables
    current_date = date.today()
    current_date = current_date.strftime("%B %d %Y")
    thread = None
    sentence_file = ''
    text_memory = ''
    speaker_name = None
    cuda_available = is_available()
    gst_dict = {}
    # preload style token dict with zeros
    for index, _ in enumerate(range(10)):
        gst_dict[str(index)] = float(0.0)

    # set icon depending on distro
    icon = PATH_GUI_ICON + '.ico' if _platform == 'Windows' else PATH_GUI_ICON + '.png'

    # set the theme
    #sg.theme('DarkTeal6')
    sg.theme('CustomTheme') # Custom theme defined at the bottom of the script. This theme can be modified or replaced with default themes, see above.

    # init default settings
    radio_keys = {'radioGL': 'GriffinLim', 'radioWR': 'WaveRNN', 'radioMG': 'MelGAN'}
    radio_image_keys = {'radioGL': 'imgradioGL', 'radioWR': 'imgradioWR', 'radioMG': 'imgradioMG'}
    selected_color = ('white', '#273c75')
    active_radio_button = 'radioGL'
    loadingAnimation = sg.Image(filename=LOADING_GIF, size=(10, 1), visible=False, key='loadingAnim', background_color='white')
    use_cuda = sg.Checkbox('Use CUDA?', default=cuda_available, font=('Arial', 11), visible=cuda_available, key='use_cuda')
    cuda_color = 'green' if cuda_available else 'red'
    cuda_text = '(CUDA Supported)' if cuda_available else '(CUDA Not Supported)'
    generate_button_disabled = False

    # get project folders and speaker_list if multispeaker model
    project_folders = [Path(folder) for folder in glob(str(Path(PATH_PROJECT, '*/')))]
    if project_folders:
        CURRENT_PROJECT_PATH = str(project_folders[0])
        project_folder_name = [str(folder.name) for folder in project_folders]
        speakers_file_path = Path(CURRENT_PROJECT_PATH, "speakers.json")
        if speakers_file_path.is_file():
            with open(speakers_file_path, 'r') as json_file:
                speaker_data = json.load(json_file)
            speaker_list = [speaker for speaker, _ in speaker_data.items()]
        else:
            speaker_list = ['Default']
    else:
        CURRENT_PROJECT_PATH = ''
        generate_button_disabled = True
        project_folder_name = ['No Projects']
        speaker_list = ['No Speaker']
        print(f'[!] No model found in projects folder: {PATH_PROJECT}')

    max_length_name = len(max(speaker_list, key=len)) + 2

    # All the stuff inside your window.
    layout = [
        [sg.Text('Project Settings:', font=('Arial', 12, 'bold'))],
        [sg.Text(cuda_text, text_color=cuda_color, font=('Arial', 11, 'bold')), use_cuda],
        [sg.Text('Project:', pad=[5, 5], size=(10, 1), justification='left', font=('Arial', 11), key='lblProject'),
         sg.DropDown(project_folder_name, project_folder_name[0], enable_events=True, pad=[5, 5], font=('Arial', 11), key='dbProject')],
        [sg.Text('Speaker:', pad=[5, 5], size=(10, 1), justification='left', font=('Arial', 11), key='lblSpeaker'),
         sg.DropDown(speaker_list, speaker_list[0], size=(max_length_name, 1), pad=[5, 5], font=('Arial', 11), key='dbSpeaker'),
         sg.Checkbox('Generate for all', default=False, font=('Arial', 11), key='create_all')],

        [sg.Text('_' * 90)],

        [sg.Text('Vocoder Settings:', font=('Arial', 12, 'bold'))],
        [sg.Image(data=CHECK_ICON, pad=(0, 5), key='imgradioGL'), sg.Button('GriffinLim', size=(12, 1), pad=((0, 15), 5), key='radioGL'),
         sg.Image(data=UNCHECK_ICON, pad=(0, 5), key='imgradioWR'), sg.Button('WaveRNN', size=(12, 1), pad=((0, 15), 5), key='radioWR'),
         sg.Image(data=UNCHECK_ICON, pad=(0, 5), key='imgradioMG'), sg.Button('MelGAN', size=(12, 1), pad=((0, 15), 5), key='radioMG')
         ],

        [sg.Text('_' * 90)],

        [sg.Text('Emotion Settings: ', font=('Arial', 12, 'bold'))],
        [sg.Text('Emotion: ', size=(10, 1), font=('Arial', 11)), sg.DropDown(['Normal', 'Angry', 'Dominant', 'Calm'], 'Normal', font=('Arial', 11), enable_events=True, key='dbEmotion')],
        [sg.Text('Speed of Speech: ', pad=[5, 5], size=(20, 1), font=('Arial', 11), key='token0'),
         sg.Slider(range=(-20, 20), default_value=0, size=(30, 10), orientation='horizontal', font=('Arial', 8, 'bold'), key='speedSlider')],
        [sg.Text('Dominance of Speech: ', pad=[5, 5], size=(20, 1), font=('Arial', 11), key='token1and2'),
         sg.Slider(range=(-20, 20), default_value=0, size=(30, 10), orientation='horizontal', font=('Arial', 8, 'bold'), key='emotionSlider')],
        [sg.Text('Tone of Speech: ', pad=[5, 5], size=(20, 1), font=('Arial', 11), key='token5'),
         sg.Slider(range=(-20, 20), default_value=0, size=(30, 10), orientation='horizontal', font=('Arial', 8, 'bold'), key='toneSlider')],

        [sg.Text('_' * 90)],

        # [sg.Checkbox('Use file for speech generation', False, font=('Arial', 11, 'bold'), enable_events=True, key='cbLoadFile')],
        [sg.Text('Use file for speech generation', font=('Arial', 11, 'bold'), key='lblLoadFile'), 
         sg.Button('Browse', size=(8, 1), pad=(1, 5), key='btnFileBrowse'),
         sg.Button(image_data=EXIT_ICON, pad=(0, 5), visible=False, button_color=('black', '#ff5e5e'), key='btnCloseFileLoad'),],
        [sg.Multiline('Im Minental versammelt sich eine Armee des Bösen unter der Führung von Drachen! Wir müssen sie aufhalten, so lange wir noch können.', 
                      size=(65, 6), pad=[5, 5], border_width=1, font=('Arial', 12), text_color=TEXT_COLOR, background_color=TEXTINPUT_BACKGROUND, key='textInput'),
         sg.Button('Output\nFolder', size=(5, 3), enable_events=True, key='btnOpenOutput')],
        [sg.Button('Generate', size=(12, 1), disabled=generate_button_disabled, key='btnGenerate'), sg.Button('Exit', size=(8, 1), key='btnExit'), loadingAnimation]
    ]

    # Create the Window
    window = sg.Window('Mozilla TTS (Generate Speech From Text)', icon=icon, layout=layout, finalize=True)
    window.FindElement('btnGenerate').Widget.config(activebackground='#273c75', activeforeground='white')
    window.FindElement('btnExit').Widget.config(activebackground='#273c75', activeforeground='white')
    window.FindElement('btnOpenOutput').Widget.config(activebackground='#273c75', activeforeground='white')
    window[active_radio_button].update(button_color=selected_color)
    for key, _ in radio_keys.items():
        window.FindElement(key).Widget.config(activebackground='#273c75', activeforeground='white')


    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read(timeout=50)

        if event in ('btnExit', 'Exit', None):  # if user closes window or clicks exit
            break

        # if another porject is select, load the corresponding configuration files
        if event in 'dbProject':
            CURRENT_PROJECT_PATH = str(Path(PATH_PROJECT, values['dbProject']))
            speaker_file_path = Path(CURRENT_PROJECT_PATH, "speakers.json")
            if speaker_file_path.is_file():
                with open(speaker_file_path, 'r') as json_file:
                    speaker_data = json.load(json_file)
                speaker_list = [speaker for speaker, _ in speaker_data.items()]
                max_length_name = len(max(speaker_list, key=len)) + 2
                window['dbSpeaker'].update(values=speaker_list)
                window['dbSpeaker'].set_size(size=(max_length_name, None))
            else:   
                window['dbSpeaker'].update(values=["Default"])
                window['dbSpeaker'].set_size(size=(len("Default"), None))

        # select emotion weights from dropdown gui element
        if event in 'dbEmotion':
            emotions = get_emotion_weights(values['dbEmotion'].lower())
            for k, v in emotions.items():
                window[k].update(v)

        if event in radio_keys:
            for k in radio_keys:
                window[k].update(button_color=sg.theme_button_color())
            window[event].update(button_color=selected_color)
            for key, value in radio_image_keys.items():
                if key == event:
                    window[value].update(data=CHECK_ICON)
                else:
                    window[value].update(data=UNCHECK_ICON)
            active_radio_button = event

        # open a prompt if the user wants to generate speech from an input file, textbox will be disabled
        if event in 'btnFileBrowse':
            path_to_textfile = sg.PopupGetFile('Please select a file or enter the file name', default_path=ROOT_PATH, initial_folder=ROOT_PATH,
                                               icon='g.ico', no_window=True, keep_on_top=True, file_types=(('Text file', '.txt'),))
            if path_to_textfile:
                sentence_file = path_to_textfile
                text_memory = window['textInput'].get()
                window['textInput'].update(disabled=True, background_color='#a7a5a5', value=f'Using textfile for speech generation: {path_to_textfile}')
                window['btnCloseFileLoad'].update(visible=True)

        if event in 'btnCloseFileLoad':
            window['btnCloseFileLoad'].update(visible=False)
            sentence_file = ''
            window['textInput'].update(disabled=False, background_color=TEXTINPUT_BACKGROUND)
            window['textInput'].update(text_memory)

        # open the output folder of the currently selected speaker
        if event in 'btnOpenOutput':
            if Path(CURRENT_PROJECT_PATH, 'output', current_date, values['dbSpeaker']).is_dir():
                speaker_path = Path(CURRENT_PROJECT_PATH, 'output', current_date, values['dbSpeaker'])
                open_output_folder(speaker_path=speaker_path)
            else:
                default_path = Path(CURRENT_PROJECT_PATH, 'output')
                open_output_folder(speaker_path=default_path)

        # stop synthezising speech
        if event in 'btnGenerate' and thread:
            if values['create_all']:
                stop_threads = True

        # start synthezising speech from the input
        if event in 'btnGenerate' and not thread:
            text = values['textInput'].replace('\n', '')
            if text or sentence_file:
                if values['dbSpeaker'] in 'Default':
                    speakers_file = ''
                    speaker_name = "Default"
                else:
                    speakers_file = Path(CURRENT_PROJECT_PATH, 'speakers.json')
                    speaker_name = values['dbSpeaker']

                if values['create_all']:
                    window['btnGenerate'].update(text='Cancel', button_color=('white', '#ff5e5e'))
                speaker = speaker_list if values['create_all'] else [speaker_name]
                vocoder = [val for key, val in radio_keys.items() if active_radio_button == key][0]

                gst_dict['0'] = round(float(values['speedSlider'] / 100), 5)
                emotion_temp = round(float(values['emotionSlider'] / 100), 5)
                gst_dict['1'] = emotion_temp
                if values['emotionSlider'] > 0:
                    gst_dict['2'] = round(emotion_temp - 0.010, 5)
                elif values['emotionSlider'] < 0:
                    gst_dict['2'] = round(emotion_temp + 0.010, 5)
                else:
                    gst_dict['2'] = emotion_temp
                gst_dict['5'] = round(float(values['toneSlider'] / 100), 5)

                # run speech generation in a new thread
                thread = threading.Thread(target=synthesize_tts,
                                          args=(text,
                                                values['use_cuda'],
                                                True, # use gst
                                                gst_dict,
                                                CURRENT_PROJECT_PATH,
                                                speaker_list,
                                                speaker,
                                                str(vocoder),
                                                sentence_file), daemon=True)
                thread.start()
                loadingAnimation.Update(filename=LOADING_GIF, visible=True)
            else:
                sg.Popup('Type something into the textbox or select a file to generate speech.', title='Missing input!',
                         line_width=65, icon='g.ico')
                window['textInput'].SetFocus()

        if status:  # If gen process finish -> stop loading.gif and open output folder
            print('Finished')
            loadingAnimation.Update(filename=LOADING_GIF, visible=False)
            if Path(CURRENT_PROJECT_PATH, 'output').is_dir():
                speaker_path = Path(CURRENT_PROJECT_PATH, 'output', current_date, speaker_name)
                open_output_folder(speaker_path=speaker_path)
            status = False


        if thread:  # If thread is running display loading gif
            loadingAnimation.UpdateAnimation(source=LOADING_GIF, time_between_frames=100)
            thread.join(timeout=0)
            if not thread.is_alive():       # the thread finished
                loadingAnimation.Update(filename=LOADING_GIF, visible=False)
                window['btnGenerate'].update(text='Generate', button_color=('white', '#40739e'))
                thread = None               # reset variable for next run

    window.close()


if __name__ == '__main__':
    # Paths
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    if ROOT_PATH:
        os.chdir(ROOT_PATH)
    MEDIA_PATH = str(Path(ROOT_PATH, "media"))
    LOADING_GIF = str(Path(MEDIA_PATH, "loading.gif"))
    PATH_GUI_ICON = str(Path(MEDIA_PATH, "g"))
    PATH_PROJECT = str(Path(ROOT_PATH, "Trainings/"))

    # Icons base64 encoded
    EXIT_ICON = 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAAtElEQVRIie2VTQ4CIQyFv3gHEo/gQu/hxDMT7yAL9zNzj3ExJTGVASkhLpyXNOGnfa8FUmBHJW7ADCxGm4AhJzA1kEcbcwLRyYqP+EMD2VcoCZyAO3BM7DnAA5caQV2il3kQwnfyIHs+E18UcMBD1p6slaTWzAI626DGTvmaBHQlqcw347u/omIGdD6i7pfc/ZmehUAfRRTx4mMWqMXve9GsMqk1WFv+Jgba/oQRuBoK/We8AFWjkoBMewj/AAAAAElFTkSuQmCC'
    CHECK_ICON = 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAAnklEQVRIie2VMQ6AIAxFX7yDxiN5FT2uDu56EB2EiFVSCg4OvoSJ9r+kBIAfIx2wAlvmWlxGlKUgPJRE8UW53PqrgrAkPi9ogN7SYDmDBphc/ZDanyqogdHVzkD7piAWniXoOUbhCccyiT2zYBBBWrhZIAO1cLMArvN+mnmxIJRo4dkCOM9A47V7kNwvn4pVFFpXmPFIR9mfoH44Pzd2G42KEKyvKj8AAAAASUVORK5CYII='
    UNCHECK_ICON = 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAAUUlEQVRIiWNgGAUkAg8GBobHDAwM/8nEj6Bm4ASPKDAc2RKcAKaIXIChn4kCw4gCoxaMWjBqwagFg9GCx1Ca3KIa2QyswIOBsjqBYIUzCjAAALrIRhJJbcctAAAAAElFTkSuQmCC'

    # Theme settings
    BACKGROUND_COLOR = '#dcdde1'
    TEXT_COLOR = '#24292E'
    BUTTON_COLOR = '#40739e'
    PROGESS_COLOR = '#273c75'
    TEXTINPUT_BACKGROUND = '#f5f6fa'
    sg.LOOK_AND_FEEL_TABLE['CustomTheme'] = {
        'BACKGROUND': BACKGROUND_COLOR,
        'TEXT': TEXT_COLOR,
        'INPUT': TEXTINPUT_BACKGROUND,
        'TEXT_INPUT': TEXT_COLOR,
        'SCROLL': PROGESS_COLOR,
        'BUTTON': ('white', BUTTON_COLOR),
        'PROGRESS': (PROGESS_COLOR, '#D0D0D0'),
        'BORDER': 1, 'SLIDER_DEPTH': 2, 'PROGRESS_DEPTH': 0,
        }

    main_gui()
