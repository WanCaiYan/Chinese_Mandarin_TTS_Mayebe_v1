import os
import wave
from datetime import datetime
import io
import numpy as np
import tensorflow as tf
from datasets import audio
from infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence, sequence_to_text
import time
import h5py


class Synthesizer:
	def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
		log('Constructing model: %s' % model_name)
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
			if hparams.predict_linear and not gta:
				self.linear_outputs = self.model.linear_outputs
				self.linear_spectrograms = tf.placeholder(tf.float32, (None, hparams.num_freq), name='linear_spectrograms')
				self.linear_wav_outputs = audio.inv_spectrogram_tensorflow(self.linear_spectrograms, hparams)

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

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPU as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, speaker_id_list, basenames, out_dir, log_dir, mel_filenames):
		start = time.clock()
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		for text in texts:
			print(text)
			# text_list = text.split('_')  # [q][ing1][ ][h][ua2][ ][d][a4][ ][x][ue2]
			# text_list_join = ''.join(text_list)  # qing1 hua2 da4 xue2
			# text_seq = text_to_sequence(text, cleaner_names)  # 1 2 3 4 5 6 ....
			# text_seq_text = sequence_to_text(text_seq)  # qing1 hua2 da4 xue2~

			# print('LOG--------------------text from metadata:', text, len(text))
			# print('LOG--------------------text list:', str(text_list), len(text_list))
			# print('LOG--------------------text people like:', text_list_join, len(text_list_join))
			# print('LOG--------------------seq has ~:', str(text_seq), len(text_seq))
			# print('LOG--------------------reverse text has ~:', text_seq_text, len(text_seq_text))
			# assert len(text_seq_text) == len(text_list_join) + 1 and len(text_seq) == len(text_list) + 1


		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_speaker_id: speaker_id_list,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		if self.gta:
			np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
			target_lengths = [len(np_target) for np_target in np_targets]
			padded_targets = self._prepare_targets(np_targets, self._hparams.outputs_per_step)
			feed_dict[self.model.mel_targets] = padded_targets.reshape(len(np_targets), -1, hparams.num_mels)

		if self.gta or not hparams.predict_linear:
			tf_run_start = time.time()
			mels, alignments, stop_tokens = self.session.run([self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
			tf_run_end = time.time()
			print('tf_run need time:', tf_run_end - tf_run_start)
			# if self.gta:
			# 	mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)] #Take off the reduction factor padding frames for time consistency with wavenet
			# 	assert len(mels) == len(np_targets)
			target_lengths = self._get_output_lengths(stop_tokens)

			#Take off the batch wise padding
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
			assert len(mels) == len(texts)

		else:
			linears, mels, alignments, stop_tokens = self.session.run([self.linear_outputs, self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)

			#Get Mel/Linear lengths for the entire batch from stop_tokens predictions
			target_lengths = self._get_output_lengths(stop_tokens)

			#Take off the batch wise padding
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
			linears = [linear[:target_length, :] for linear, target_length in zip(linears, target_lengths)]
			assert len(mels) == len(linears) == len(texts)

		# if basenames is None:
		# 	#Generate wav and read it
		# 	wav = audio.inv_mel_spectrogram(mels.T, hparams)
		# 	audio.save_wav(wav, 'temp.wav', hparams) #Find a better way

		# 	chunk = 512
		# 	f = wave.open('temp.wav', 'rb')
		# 	p = pyaudio.PyAudio()
		# 	stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
		# 		channels=f.getnchannels(),
		# 		rate=f.getframerate(),
		# 		output=True)
		# 	data = f.readframes(chunk)
		# 	while data:
		# 		stream.write(data)
		# 		data=f.readframes(chunk)

		# 	stream.stop_stream()
		# 	stream.close()

		# 	p.terminate()
		# 	return
		
		# elapsed = (time.clock() - start)
		# print("---------------------NN Model Time used:",elapsed)
			
		# if basenames is None:
		# 	#Generate wav and read it
		# 	#wav = audio.inv_mel_spectrogram(mels[0].T, hparams)
		# 	#elapsed = (time.clock() - start)
		# 	#print("---------------------Inv 1 Time used:",elapsed)
		# 	#audio.save_wav(wav, 'temp.wav', hparams) #Find a better way
		# 	#elapsed = (time.clock() - start)
		# 	#print("---------------------Save 1 Time used:",elapsed)
		# 	#save wav (linear -> wav)
		# 	linear_wav = self.session.run(self.linear_wav_outputs, feed_dict={self.linear_spectrograms: linears[0]})
		# 	elapsed = (time.clock() - start)
		# 	print("---------------------NN Linera Time used:",elapsed)
			
		# 	wav = audio.inv_preemphasis(linear_wav, hparams.preemphasis)
		# 	audio.save_wav(wav, 'temp-linear.wav', hparams)
			
		# 	elapsed = (time.clock() - start)
		# 	print("---------------------Time used:",elapsed)
		# 	print('---------------------It is finished')
		# 	return


		saved_mels_paths = []
		speaker_ids = []
		# print('????????????????????????????????????', len(mels))
		
		for i in range(len(mels)):
			# print('i is', i)
			mel = mels[i]
			# linear = linears[i]

			speaker_id = '<no_g>'
			speaker_ids.append(speaker_id)
			mel_filename = os.path.join(out_dir, 'mel-{}.h5'.format(basenames[i]))
			# np.save(mel_filename, mel.T, allow_pickle=False)
			# np.save(mel_filename, mel.T, allow_pickle=False)
			# mel
			f = h5py.File(mel_filename,'w')   #创建一个h5文件，文件指针是f
			f['feats'] = mel                 #将数据写入文件的主键data下面
			f.close()   


			# pertrain: norm1 -> zhao: norm2_wavegan_v1
			wavegan_v1_out_dir = out_dir + '_for_wavegan_v1'
			os.makedirs(wavegan_v1_out_dir, exist_ok=True)
			mel_wavegan_v1_filename = os.path.join(wavegan_v1_out_dir, 'mel-{}.h5'.format(basenames[i]))
			tmp_a = h5py.File('wavegan_v1_stats.h5', 'r')
			ma = tmp_a['mean'][:]
			sa = tmp_a['scale'][:]
			tmp_b = h5py.File('zhao_wavegan_v1_stats.h5', 'r')
			mb = tmp_b['mean'][:]
			sb = tmp_b['scale'][:]
			mel_wavegan_v1 = mel * sa + ma
			mel_wavegan_v1 = (mel_wavegan_v1 - mb) / sb
			f = h5py.File(mel_wavegan_v1_filename,'w')   #创建一个h5文件，文件指针是f
			f['feats'] = mel_wavegan_v1                 #将数据写入文件的主键data下面
			f.close()   

			# pretrain: norm1 -> zhao: norm2_melgan_v1
			melgan_v1_out_dir = out_dir + '_for_melgan_v1'
			os.makedirs(melgan_v1_out_dir, exist_ok=True)
			mel_melgan_v1_filename = os.path.join(melgan_v1_out_dir, 'mel-{}.h5'.format(basenames[i]))
			tmp_a = h5py.File('wavegan_v1_stats.h5', 'r')
			ma = tmp_a['mean'][:]
			sa = tmp_a['scale'][:]
			tmp_b = h5py.File('zhao_melgan_v1_stats.h5', 'r')
			mb = tmp_b['mean'][:]
			sb = tmp_b['scale'][:]
			mel_melgan_v1 = mel * sa + ma
			mel_melgan_v1 = (mel_melgan_v1 - mb) / sb
			f = h5py.File(mel_melgan_v1_filename,'w')   #创建一个h5文件，文件指针是f
			f['feats'] = mel_melgan_v1                 #将数据写入文件的主键data下面
			f.close()   


			# linear_filename = os.path.join(out_dir, 'linear-timeFirst-{}.npy'.format(basenames[i]))
			# np.save(linear_filename, linear, allow_pickle=False)

			saved_mels_paths.append(mel_filename)
			#save wav (mel -> wav)
			# wav = audio.inv_mel_spectrogram(mel.T, hparams)
			# audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])), hparams)
			

			#save alignments
			plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{}.png'.format(basenames[i])),
			info='{}'.format('#####'), split_title=True)

			#save mel spectrogram plot
			plot.plot_spectrogram(mel, os.path.join(log_dir, 'plots/mel-{}.png'.format(basenames[i])),
			info='{}'.format(texts[i]), split_title=True)
			
			# print('linear', i)
			#save wav (linear -> wav)
			# linear_wav = self.session.run(self.linear_wav_outputs, feed_dict={self.linear_spectrograms: linears[i]})
			# wav = audio.inv_preemphasis(linear_wav, hparams.preemphasis)
			# audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-linear.wav'.format(basenames[i])), hparams)
			
			#save mel spectrogram plot
			# plot.plot_spectrogram(linears[i], os.path.join(log_dir, 'plots/linear-{}.png'.format(basenames[i])),
			# info='{}'.format(texts[i]), split_title=True, auto_aspect=True)
		'''
		for i, mel in enumerate(mels):
			#Get speaker id for global conditioning (only used with GTA generally)
			speaker_id = '<no_g>'
			speaker_ids.append(speaker_id)

			# Write the spectrogram to disk
			# Note: outputs mel-spectrogram files and target ones have same names, just different folders
			mel_filename = os.path.join(out_dir, 'mel-{}.npy'.format(basenames[i]))
			np.save(mel_filename, mel.T, allow_pickle=False)
			saved_mels_paths.append(mel_filename)

			if log_dir is not None:
				#save wav (mel -> wav)
				wav = audio.inv_mel_spectrogram(mel.T, hparams)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])), hparams)

				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{}.png'.format(basenames[i])),
					info='{}'.format(texts[i]), split_title=True)

				#save mel spectrogram plot
				plot.plot_spectrogram(mel, os.path.join(log_dir, 'plots/mel-{}.png'.format(basenames[i])),
					info='{}'.format(texts[i]), split_title=True)

				if hparams.predict_linear and not self.gta:
					
					print('linear', i)
					#save wav (linear -> wav)
					linear_wav = self.session.run(self.linear_wav_outputs, feed_dict={self.linear_spectrograms: linears[i]})
					wav = audio.inv_preemphasis(linear_wav, hparams.preemphasis)
					audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-linear.wav'.format(i)), hparams)

					#save mel spectrogram plot
					plot.plot_spectrogram(linears[i], os.path.join(log_dir, 'plots/linear-{}.png'.format(basenames[i])),
						info='{}'.format(texts[i]), split_title=True, auto_aspect=True)
					i += 1
		'''
		# print('saved_mels_paths', saved_mels_paths, speaker_ids)
		return saved_mels_paths, speaker_ids

	def eval(self, batch):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in batch]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		linears, stop_tokens = self.session.run([self.linear_outputs, self.stop_token_prediction], feed_dict=feed_dict)

		#Get Mel/Linear lengths for the entire batch from stop_tokens predictions
		target_lengths = self._get_output_lengths(stop_tokens)

		#Take off the batch wise padding
		linears = [linear[:target_length, :] for linear, target_length in zip(linears, target_lengths)]
		assert len(linears) == len(batch)

		#save wav (linear -> wav)
		results = []
		for i, linear in enumerate(linears):
			linear_wav = self.session.run(self.linear_wav_outputs, feed_dict={self.linear_spectrograms: linear})
			wav = audio.inv_preemphasis(linear_wav, hparams.preemphasis)
			results.append(wav)
		return np.concatenate(results)


	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		return np.stack([self._pad_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _get_output_lengths(self, stop_tokens):
		#Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
		output_lengths = [row.index(1) + 1 if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
		return output_lengths
