#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import sys
sys.path.append('/home/hujk17/TTS/WaveGAN')
sys.path.append('/home/hujk17/TTS/Taco-2-Rayhane-WaveGan-Norm')
sys.path.append(os.path.dirname(os.path.abspath('__file__')))
from CrystalTTS.demo.CrystalTTS import CrystalTTS
import shutil
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

import parallel_wavegan.models

from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.utils import read_hdf5

import random
import jieba

# tacotron-2
import tensorflow as tf
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence, sequence_to_text
import h5py
from hparams import hparams

def hanzi_pause2phoneme_pause(c2p, hanzi_pause):
    # global ASR_P_cnt, CI_P_cnt, ASR_CI_P_cnt, ASR_notCI_P_cnt
    # global ASR_notCI_P_sentences
    ASR_P_cnt = 0
    CI_P_cnt = 0
    ASR_CI_P_cnt = 0
    ASR_notCI_P_cnt = 0
    # 我的母亲和许多别的人#3都这样称唿她#4
    hanzi = hanzi_pause.replace('#', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '')
    hanzi_jieba = jieba.cut(hanzi, cut_all=False)
    hanzi_jieba_pause_list = []
    for x in hanzi_jieba:
        for s in x:
            hanzi_jieba_pause_list.append(s)
        hanzi_jieba_pause_list.append('@')
    Crystal_file_name = 'child0.2_' + time.strftime('%Y-%m-%d%H-%M-%S_', time.localtime(time.time())) + str(
        random.randint(100000, 999999)) + '.txt'
    tag_ans = c2p.textAnalyze(hanzi, Crystal_file_name)
    assert tag_ans
    with open(Crystal_file_name, 'r', encoding='utf-8-sig') as f:
        raw_phoneme = f.read().strip()
    if os.path.exists(Crystal_file_name):
        os.remove(Crystal_file_name)
    # print('CI:', raw_phoneme)
    print('jieba:', ''.join(hanzi_jieba_pause_list))

    # now is: uo3|d-e5|@m-u3|q-in1|@h-e2|x-v3|d-uo1|@b-ie2|d-e5|r-en2|@d-ou1|zh-e4|iang4|ch-eng1|@h-u1|t-a1|@。
    phoneme = raw_phoneme.replace('。', '').replace('|@', '|@|')[:-1]
    phoneme_list = phoneme.split('|')
    while '@' in phoneme_list:
        phoneme_list.remove('@')

    hanzi_list = list(hanzi_pause.replace('#1', '^').replace('#2', '/').replace('#3', '@').replace('#4', '#'))
    print('hanzi:', hanzi)
    print('phoneme_list:', phoneme_list, '|')
    print(len(phoneme_list), len(hanzi))
    if len(phoneme_list) != len(hanzi):
        er_cnt = hanzi_list.count('儿')
        if er_cnt > 1:
            return False, False

        # print('-----', phoneme_list)
        # print(len(phoneme_list), er_cnt, len(hanzi))
        assert len(phoneme_list) + er_cnt == len(hanzi)
        while '儿' in hanzi_list:
            hanzi_list.remove('儿')
        assert '儿' not in hanzi_list
        while '儿' in hanzi_jieba_pause_list:
            hanzi_jieba_pause_list.remove('儿')
        assert '儿' not in hanzi_jieba_pause_list
        # print('------------------------------------: eeeeeeeeeeeeer', )
    else:
        assert len(phoneme_list) == len(hanzi)


    pause_symbol = ['^', '/', '@', '#']
    phoneme_withHanziPause = []
    j_jieba = 0
    k_phoneme = 0
    # [uo3, @, d-e5, @, h-a1, @]
    # [我, 的, /, 哈, #]
    # print(phoneme_list)
    # print(hanzi_list)
    for i, x in enumerate(hanzi_list):
        if x in pause_symbol:
            # phoneme_withHanziPause.append(x)
            ASR_P_cnt += 1
            if hanzi_jieba_pause_list[j_jieba] == '@':
                phoneme_withHanziPause.append(x)
                j_jieba += 1
                ASR_CI_P_cnt += 1
                CI_P_cnt += 1
            else:
                if x == '^':
                    continue
                else:
                    phoneme_withHanziPause.append(x)
                    ASR_notCI_P_cnt += 1
        else:
            if hanzi_jieba_pause_list[j_jieba] == '@':
                j_jieba += 1
                CI_P_cnt += 1
            assert x == hanzi_jieba_pause_list[j_jieba]
            j_jieba += 1
            # print('tmp', hanzi_list)
            # print('tmp', k_phoneme, len(phoneme_list), phoneme_list)
            phoneme_withHanziPause.append(phoneme_list[k_phoneme])
            k_phoneme += 1

    assert k_phoneme == len(phoneme_list) 
    assert j_jieba == len(hanzi_jieba_pause_list)
    # [uo3, d-e5, /, h-a1, #]
    phoneme_withHanziPause_str = '-'.join(phoneme_withHanziPause).replace('1', '-1').replace('2', '-2').replace('3', '-3').replace('4', '-4').replace('5', '-5').replace('6', '-6')
    phoneme_withHanziPause_str = phoneme_withHanziPause_str.replace('-', '_')
    # print('ASR  :', phoneme_withHanziPause_str)
    # print('ASR Pause:', ASR_P_cnt, 'CI Pause', CI_P_cnt, 'both', ASR_CI_P_cnt, 'ASR not CI', ASR_notCI_P_cnt)
    # if ASR_notCI_P_cnt > 0:
    #     ASR_notCI_P_sentences += 1
    return phoneme_withHanziPause_str, ''.join(hanzi_list)


# phoneme -> mel
class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_speaker_id = tf.placeholder(tf.int32, [None], 'input_speaker_id')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
        targets = tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets')
        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)
            if gta:
                self.model.initialize(inputs, input_speaker_id, input_lengths, targets, gta=gta)
            else:
                self.model.initialize(inputs, input_speaker_id, input_lengths)
            self.alignments = self.model.alignments
            self.mel_outputs = self.model.mel_outputs
            self.stop_token_prediction = self.model.stop_token_prediction

        self.gta = gta
        self._hparams = hparams
        #pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        #explicitely setting the padding to a value that doesn't originally exist in the spectogram
        #to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -(hparams.max_abs_value + .1)
        else:
            self._target_pad = -0.1

        #Memory allocation on the GPU as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


    def synthesize(self, texts, speaker_id_list, ma, sa, mb, sb):
        # start = time.clock()
        hparams = self._hparams
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        for text in texts:
            print(text)
            text_list = text.split('_')  # [q][ing1][ ][h][ua2][ ][d][a4][ ][x][ue2]
            text_list_join = ''.join(text_list)  # qing1 hua2 da4 xue2
            text_seq = text_to_sequence(text, cleaner_names)  # 1 2 3 4 5 6 ....
            text_seq_text = sequence_to_text(text_seq)  # qing1 hua2 da4 xue2~

            print('LOG--------------------text from metadata:', text, len(text))
            print('LOG--------------------text list:', str(text_list), len(text_list))
            print('LOG--------------------text people like:', text_list_join, len(text_list_join))
            print('LOG--------------------seq has ~:', str(text_seq), len(text_seq))
            print('LOG--------------------reverse text has ~:', text_seq_text, len(text_seq_text))
            assert len(text_seq_text) == len(text_list_join) + 1 and len(text_seq) == len(text_list) + 1


        seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
        
        input_lengths = [len(seq) for seq in seqs]
        seqs = self._prepare_inputs(seqs)
        
        feed_dict = {
            self.model.inputs: seqs,
            self.model.input_speaker_id: speaker_id_list,
            self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }


        tf_run_start = time.time()
        mels, alignments, stop_tokens = self.session.run([self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
        # tf_run_end = time.time()
        # print('tf_run need time:', tf_run_end - tf_run_start)
        target_lengths = self._get_output_lengths(stop_tokens)

        #Take off the batch wise padding
        mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
        assert len(mels) == len(texts)
        denorm_mels = []
        for i in range(len(mels)):
            mel = mels[i]
            mel_wavegan_v1 = mel * sa + ma
            mel_wavegan_v1 = (mel_wavegan_v1 - mb) / sb
            denorm_mels.append(mel_wavegan_v1)
        return denorm_mels


    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs])

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _get_output_lengths(self, stop_tokens):
        #Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
        output_lengths = [row.index(1) + 1 if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
        return output_lengths
    

def main():
    time_start = time.time()
    #[1] hanzi -> phoneme Load
    c2p = CrystalTTS(dllpath=u'Load_Then_Predict/CrystalTTS/demo/CrystalDll.so')
    c2p.initialize(textpath=u'Load_Then_Predict/CrystalTTS/data/putonghua/text')

    #[2] phoneme -> mel, Load 1: denorm by h5py std File
    tmp_a = h5py.File('/home/hujk17/TTS/Taco-2-Rayhane-WaveGan-Norm/wavegan_v1_stats.h5', 'r')
    ma = tmp_a['mean'][:]
    sa = tmp_a['scale'][:]
    tmp_b = h5py.File('/home/hujk17/TTS/Taco-2-Rayhane-WaveGan-Norm/zhao_wavegan_v1_stats.h5', 'r')
    mb = tmp_b['mean'][:]
    sb = tmp_b['scale'][:]

    #[2] phoneme -> mel, Load 2: tacotron model
    taco_chekpoint_dir = '/home/hujk17/TTS/Taco-2-Rayhane-WaveGan-Norm/logs-Tacotron/taco_pretrained'
    taco_checkpoint_path = tf.train.get_checkpoint_state(taco_chekpoint_dir).model_checkpoint_path
    synth = Synthesizer()
    synth.load(taco_checkpoint_path, hparams)
    print('run finish 222222')

    #[3] mel -> wav, Load torch model
    # dumpdir = '/home/hujk17/TTS/Load_Then_Predict/tacotron_output/eval_for_wavegan_v1'
    wavegan_checkpoint_path = '/home/hujk17/TTS/WaveGAN/selftrain_zhaodan_v1_long/checkpoint-1000000steps.pkl'
    dirname = os.path.dirname(wavegan_checkpoint_path)
    config_path = os.path.join(dirname, "config.yml")
    logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(wavegan_checkpoint_path, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {wavegan_checkpoint_path}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    use_noise_input = not isinstance(
        model, parallel_wavegan.models.MelGANGenerator)
    print('use noise input', use_noise_input)
    pad_fn = torch.nn.ReplicationPad1d(
        config["generator_params"].get("aux_context_window", 0))
    print('run finish 333333')

    #[4] Others Load 
    # check directory existence
    outdir = '/home/hujk17/TTS/sample_wavs'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    # mel_query = "*.h5"
    # mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
    # dataset = MelDataset(
    #         dumpdir,
    #         mel_query=mel_query,
    #         mel_load_fn=mel_load_fn,
    #         return_utt_id=True,
    #     )

    print('load model time:', time.time() - time_start)
    while True:
        hanzi_pause = input()
        if hanzi_pause == '0':
            break
        time_start = time.time()
        time_start_tot = time.time()
        #[B-1] hanzi -> phoneme_str, single sentence and then batch of single sentence
        phoneme_pause, raw_phoneme = hanzi_pause2phoneme_pause(c2p, hanzi_pause)
        sentences_batch = [phoneme_pause,]
        speaker_id_lists_batch = [1,]
        print('run finish BBBBB----11111, use time:', time.time() - time_start)
        time_start = time.time()

        #[B-2] phoneme_str -> mel
        mels = synth.synthesize(sentences_batch, speaker_id_lists_batch, ma, sa, mb, sb)
        print('run finish BBBBB----22222, use time:', time.time() - time_start)
        time_start = time.time()

        #[B-3] mel -> wav
        with torch.no_grad():
            for c in mels:
                x = ()
                if use_noise_input:
                    z = torch.randn(1, 1, len(c) * config["hop_size"]).to(device)
                    x += (z,)
                c = pad_fn(torch.from_numpy(c).unsqueeze(0).transpose(2, 1)).to(device)
                x += (c,)
                y = model(*x).view(-1).cpu().numpy()
                tmp_name = hanzi_pause[:6] + '_' + str(random.randint(100000, 999999)) + '_gen.wav'
                sf.write(os.path.join(outdir, tmp_name),
                        y, config["sampling_rate"], "PCM_16")
        print('run finish BBBBB----33333, use time:', time.time() - time_start)
        print('total use time:', time.time() - time_start_tot)


if __name__ == "__main__":
    main()