from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import random	
import time
import torch
import soundfile as sf
import jieba

    
def change_int2str(a):
    return str(a)

def index1(request):
    c2p = settings.C2P
    synth = settings.SYNTH
    model = settings.MODEL
    #synth2 = settings.SYNTH2
    print('str', 'sdsd')
    return HttpResponse("Rango says hey there world!")


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
    print(phoneme_list)
    print(hanzi_list)
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
                print('go one step')
                j_jieba += 1
                CI_P_cnt += 1
            assert x == hanzi_jieba_pause_list[j_jieba]
            print('now go', x)
            j_jieba += 1
            # print('tmp', hanzi_list)
            # print('tmp', k_phoneme, len(phoneme_list), phoneme_list)
            phoneme_withHanziPause.append(phoneme_list[k_phoneme])
            k_phoneme += 1
    if j_jieba == len(hanzi_jieba_pause_list) - 1 and hanzi_jieba_pause_list[j_jieba] == '@':
        j_jieba += 1
    print('now:', j_jieba)
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

def index(request, path):
    time_start = time.time()
    time_start_tot = time.time()

    c2p = settings.C2P
    synth = settings.SYNTH
    model = settings.MODEL
    pad_fn = settings.PAD_FN
    use_noise_input = settings.USE_NOISE_INPUT
    device = settings.DEVICE
    config = settings.CONFIG
    outdir = settings.OUTDIR
    ma = settings.MA
    sa = settings.SA
    mb = settings.MB
    sb = settings.SB

    hanzi_pause = path
    hanzi_pause = hanzi_pause.replace('1', '#1').replace('2', '#2').replace('3', '#3').replace('4', '#4')
    print('Input:', hanzi_pause, '|')
    
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
    Django_wav_filepath = os.path.join(outdir, tmp_name)
    # os.rename(TTS_filepath, Django_filepath)
    file_object = open(Django_wav_filepath, 'rb')
    content = file_object.read()
    return HttpResponse(content, content_type='audio/wav')
    
    
    # if os.path.exists(TTS_filepath) and os.path.getsize(TTS_filepath) > 30:
    	
    # else:
    # 	return HttpResponse('Rango says no')
    
    # # print(settings)
    # # print('test is:', settings.G_SYNTH)
    
    # return HttpResponse("Rango says hey there world!")




