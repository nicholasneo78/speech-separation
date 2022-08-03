import soundfile as sf
import librosa
import os
import json
from tqdm import tqdm
import logging

def convert(input_audio_path, output_audio_path, target_sr) -> None:
    speech_array, sr = librosa.load(input_audio_path, sr=None)
    speech_array_16k = librosa.resample(speech_array, orig_sr=sr, target_sr=target_sr)
    
    # overwrite the sound with the target sample rate
    sf.write(output_audio_path, speech_array_16k, target_sr, subtype='FLOAT')

if __name__ == '__main__':
    convert(input_audio_path='/speech_separation/datasets/example_libri/mms_1.wav', 
            output_audio_path='/speech_separation/datasets/example_libri/mms_1_8000.wav', 
            target_sr=8000)

            