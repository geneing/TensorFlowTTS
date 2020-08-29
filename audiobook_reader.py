
import tensorflow as tf

import yaml
import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile
import re
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

fastspeech2_config = AutoConfig.from_pretrained('pretrained/fastspeech2_config.yml')
#fastspeech2_config.max_position_embeddings = 20000
fastspeech2 = TFAutoModel.from_pretrained(
    config=fastspeech2_config,
    pretrained_path="pretrained/fastspeech2-150k.h5",
    name="fastspeech2"
)

mb_melgan_config = AutoConfig.from_pretrained('pretrained/mb.melgan_config.yml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    pretrained_path="pretrained/mb.melgan-940k.h5",
    name="mb_melgan"
)

processor = AutoProcessor.from_pretrained(pretrained_path="pretrained/ljspeech_mapper.json")


def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
    input_ids = processor.text_to_sequence(input_text)

    # text2mel part
    if text2mel_name == "TACOTRON":
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
    elif text2mel_name == "FASTSPEECH":
        mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    elif text2mel_name == "FASTSPEECH2":
        mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    else:
        raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

    # vocoder part
    if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    elif vocoder_name == "MB-MELGAN":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    else:
        raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

    if text2mel_name == "TACOTRON":
        return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
    else:
        return mel_outputs.numpy(), audio.numpy()



input_file = "/home/eugening/Neural/MachineLearning/Speech/ESPnet/my_experiments/Around_the_world_in_80_days.txt"
output_dir = "/home/eugening/Neural/MachineLearning/Speech/TensorFlowTTS/test_output/"
fs = 22050
# synthesis

import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


with open(input_file, 'r') as f:
    txt = f.read()

doc = nlp(txt)

for i, p in enumerate(doc.sents):
    try:
        input_text = p.text.replace('\n', ' ')
        print(input_text, "\n\n")
        mels, audios = do_synthesis(input_text, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN")
        scipy.io.wavfile.write('%s/%.4d.wav'%(output_dir, i+1), rate=fs, data=audios)
    except Exception as e:
        print("{} \t Failed: {}\n\n".format(e, input_text))
