import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('__file__')))
from CrystalTTS.demo.CrystalTTS import CrystalTTS
import time
import random
from tqdm import tqdm
import jieba

c2p = CrystalTTS(dllpath=u'CrystalTTS/demo/CrystalDll.so')
c2p.initialize(textpath=u'CrystalTTS/data/putonghua/text')
hanzi_id_pause_path = '../hanzi.txt'
output_csv = '../shengyunmu.txt'
output_er_drop_path = '../er_replace_ASR-1.csv.txt'

if os.path.exists(output_csv):
    os.remove(output_csv)
if os.path.exists(output_er_drop_path):
    os.remove(output_er_drop_path)

# ASR_P_cnt = 0
# CI_P_cnt = 0
# ASR_CI_P_cnt = 0
ASR_notCI_P_sentences = 0


def hanzi_pause2phoneme_pause(hanzi_pause):
    # global ASR_P_cnt, CI_P_cnt, ASR_CI_P_cnt, ASR_notCI_P_cnt
    global ASR_notCI_P_sentences
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
    if ASR_notCI_P_cnt > 0:
        ASR_notCI_P_sentences += 1
    return phoneme_withHanziPause_str, ''.join(hanzi_list)



hanzi_id_pause_file = open(hanzi_id_pause_path, 'r', encoding='UTF-8').readlines()

for i, x in enumerate(tqdm(hanzi_id_pause_file)):
    id = i + 1
    hanzi_pause = x.strip()
    if len(hanzi_pause) == 0:
        continue
    phoneme_pause, raw_phoneme = hanzi_pause2phoneme_pause(hanzi_pause)
    if phoneme_pause is False:
        # print('2 er is hard:', id, hanzi_pause)
        with open(output_er_drop_path, 'a', encoding='utf-8') as f:
            f.writelines(id + '|' + hanzi_pause + '\n')
        continue

    with open(output_csv, 'a', encoding='utf-8') as f:
            f.writelines(phoneme_pause + '|' + '1' + '\n')
    

# print('ASR Pause:', ASR_P_cnt, 'CI Pause', CI_P_cnt, 'both', ASR_CI_P_cnt, 'ASR not CI', ASR_notCI_P_cnt)
print('ASR not CI sentences', ASR_notCI_P_sentences)
