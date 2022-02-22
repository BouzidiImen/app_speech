from PIL import Image
from array import array
import wave
from speechbrain.pretrained import EncoderASR
import streamlit as st
import pyaudio

# Global variables
chunk = 1024  # Each chunk will consist of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2  # Number of audio channels
fs = 44100  # Record at 44100 samples per second
THRESHOLD = 500


def is_silent(snd_data, threshold):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold


def record_aud(time_in_sec, record):
    if record:
        st.write('Please speak...')
        filename = "filename.wav"
        # generate file names based on input of doctor's name or ask
        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        # Open a Stream with the values we just defined
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
        frames = []  # Initialize array to store frames
        silence = 0
        not_silence = 0
        for i in range(0, int(fs / chunk * time_in_sec)):
            data = stream.read(chunk)
            frames.append(data)
            snd_data = array('h', data)
            silent = is_silent(snd_data, THRESHOLD)
            if silent and not_silence < 20:
                silence += 1
                not_silence = 0
            if silent == 0 and silence < 20:
                not_silence += 1
                silence = 0
            if silence > 180:
                stream.stop_stream()
                stream.close()
                p.terminate()
                break
        st.write('Finished Recording')
        # Open and Set the data of the WAV file
        file = wave.open(filename, 'wb')
        file.setnchannels(channels)
        file.setsampwidth(p.get_sample_size(sample_format))
        file.setframerate(fs)
        # Write and Close the File
        file.writeframes(b''.join(frames))
        file.close()
        audio_file = open(filename, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-fr",
                                            savedir="pretrained_models/asr-wav2vec2-commonvoice-fr")
        # asr_model = EncoderASR.from_hparams(source="pretrained_models/asr-wav2vec2-commonvoice-fr")
        text = st.text_area("Predicted Text", asr_model.transcribe_file(filename), key=None, height=400)
        # output text in bloc note with filename customizable
        # Add button to submit not enter
        if st.button("Submit"):
            st.success('Saved!')
    else:
        st.write('Click the button to record')


st.set_page_config(page_title="Speech Recognition", page_icon="\U0001f399")

number = st.number_input('Insert maximal recording time in second')
st.write('Maximum recording time will be  ', number, 'second')
# filename=st.text_input("Medical record number", value="")
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col3:
    record = st.button("\u23FA")
record_aud(number, record)
# when stopped we can add and output the output to see it
