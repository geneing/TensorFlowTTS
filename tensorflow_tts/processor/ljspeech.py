# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform preprocessing and raw feature extraction for LJSpeech dataset."""

import os
import re

import numpy as np
import soundfile as sf
from dataclasses import dataclass
from tensorflow_tts.processor import BaseProcessor

from g2p_en import g2p as grapheme_to_phonem

g2p = grapheme_to_phonem.G2p()

valid_symbols = g2p.phonemes
valid_symbols.append("SIL")
valid_symbols.append("END")

_pad = "pad"
_eos = "eos"
_unk = "unk"
_special = "-"

_punctuation = "!'(),.:;? "
_arpabet = valid_symbols

LJSPEECH_SYMBOLS = [_pad] + [_unk] + _arpabet + list(_punctuation) + list(_special)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

@dataclass
class LJSpeechProcessor(BaseProcessor):
    """LJSpeech processor."""

    mode: str = "train"
    cleaner_names: str = "english_cleaners"
    positions = {
        "wave_file": 0,
        "text": 1,
        "text_norm": 2,
    }
    train_f_name: str = "metadata.csv"

    def create_items(self):
        if self.data_dir:
            with open(
                os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
            ) as f:
                self.items = [self.split_line(self.data_dir, line, "|") for line in f]

    def split_line(self, data_dir, line, split):
        parts = line.strip().split(split)
        wave_file = parts[self.positions["wave_file"]]
        text_norm = parts[self.positions["text_norm"]]
        wav_path = os.path.join(data_dir, "wavs", f"{wave_file}.wav")
        speaker_name = "ljspeech"
        return text_norm, wav_path, speaker_name

    def setup_eos_token(self):
        return _eos

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path, dtype="float32")

        # convert text to ids
        ids = self.text_to_sequence(text)
        text_ids = np.asarray(ids, np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text):
        return self.symbols_to_ids(self.text_to_ph(text))

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        ids = []
        for s in symbols_list:
            try:
                ids.append(self.symbol_to_id[s])
            except KeyError as e: pass
        return ids

    def text_to_ph(self, text: str):
        return self.clean_g2p(g2p(text))

    def clean_g2p(self, g2p_text: list):
        data = []
        for i, txt in enumerate(g2p_text):
            if i == len(g2p_text) - 1:
                if txt != " " and txt != "SIL":
                    #data.append("@" + txt)
                    pass
                else:
                    data.append(
                        "END"
                    )  # TODO try learning without end token and compare results
                break
            data.append(txt) if txt != " " else data.append(
                "SIL"
            )  # TODO change it in inference
        return data
