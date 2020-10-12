import os
import re
import sys
import pickle
import re
from collections import Counter
import logging

sys.path.append(os.path.dirname(os.path.abspath('__file__')))
from CrystalTTS.demo.CrystalTTS import CrystalTTS
import time
import random
from tqdm import tqdm
import string
# from zhon.hanzi import punctuation

c2p = CrystalTTS(dllpath=u'CrystalTTS/demo/CrystalDll.so')
c2p.initialize(textpath=u'CrystalTTS/data/putonghua/text')

in_txt = './000001-010000.txt'
output_txt = 'synthesis_text_BZNSYP.txt'
output_crystal_txt = 'synthesis_text_BZNSYP_crystal.txt'

if os.path.exists(output_crystal_txt):
    os.remove(output_crystal_txt)
if os.path.exists(output_txt):
    os.remove(output_txt)



def change2phoneme(text, id=-1):
    if '\ufeff' in text:
        errors.append('有boom')
        print('------------------1111111111111111111------')
    Crystal_file_name = 'child0.2_' + time.strftime('%Y-%m-%d%H-%M-%S_', time.localtime(time.time())) + str(
        random.randint(100000, 999999)) + '.txt'
    tag_ans = c2p.textAnalyze(text, Crystal_file_name)
    
    with open(Crystal_file_name, 'r', encoding='utf-8-sig') as f:
        raw_phoneme = f.read().strip()
        raw_phoneme = raw_phoneme.replace('_JH/IY1', 'j-i4').replace('_T/IY1', 't-i1').replace('_D/IY1',
                                                                                               'd-i4').replace('_AA1/R',
                                                                                                               'h-ua2').replace(
            '_EY1', 'ei1').replace('_Y/UW', 'r-ua2')
    print('-------------raw:', raw_phoneme)
    
    phoneme = raw_phoneme.replace('|@', '-#-').replace('|', '- -').replace('。', '').replace('1', '-1').replace('2', '-2').replace('3', '-3').replace('4', '-4').replace('5', '-5').replace('6', '-6')
    while phoneme[-1] == '-':
        phoneme = phoneme[:-1]
    phoneme = phoneme.replace('-', '_')

    # replace # as blank, but end of sentence still use #
    phoneme = phoneme.replace('#', ' ')
    assert phoneme[-1] == ' '
    # phoneme[-1] = '#'
    phoneme = phoneme[:-1] + '#'
    
    # q_ing_1_ _h_ua_2_ _d_a_4_ _x_ue_2_#_  ha1   uo_3_ _d_e_5_#_m_u_3_ _q_in_1_#_h_e_2_ _x_v_3_ _d_uo_1_#_b_ie
    phoneme_split = phoneme.split('_')  # [q][ing][1][ ][h][ua][1][ ][d][a4][ ][x][ue2]
    phoneme_join = ''.join(phoneme_split)  # qing1 hua2 da4 xue2
    print('after split _', phoneme_split)
    print('use no thing join then :', phoneme_join)
    if os.path.exists(Crystal_file_name):
        os.remove(Crystal_file_name)
    return phoneme, raw_phoneme



phrase_regex = re.compile('#[0-9]')
remove_symbols_regex = re.compile('[0-9“”…（），：、—；]')
remove_stops_regex = re.compile('[。！？]')
has_alphabets_regex = re.compile('[a-zA-Zａ-ｚＡ-Ｚ]')
a = open(in_txt, encoding='utf-8').readlines()
a = [i.strip('\t\r\n') for i in a]
a = [a[i].split('\t') + [a[i + 1]] for i in range(0, len(a), 2)]
print(a[0])
print(a[1])
phoneme_list = {}
phoneme_dict = set(['~', ])
character_list = {}
character_dict = set(['~', ])
errors = []
cnt = 0
for i, line in enumerate(a):
    cnt += 1
    if len(has_alphabets_regex.findall(line[1])) > 0:
        errors.append('有字母')
        # assert False
    phrases = phrase_regex.findall(line[1])
    b = remove_symbols_regex.sub('', line[1]).split('#')


    key = line[0]
    pinyin = line[2].split(' ')
    if not b[-1] in '。！？':
        errors.append('末尾为其他标点')
        pass
    if b[-1] == '':
        errors.append('末尾缺少标点')
        b[-1] = '。'
    if key == '001464':
        b = [re.sub(r'([这明])儿', r'\1', i) for i in b]
    elif key == '009197':
        b = [re.sub(r'([会])儿', r'\1', i) for i in b]
    tmp = remove_stops_regex.sub('', ''.join(b))
    if len(tmp) != len(pinyin):
        if tmp.count('儿') == len(tmp) -len(pinyin):
            b = [i.replace('儿', '') for i in b]
        else:
            errors.append('音节数量不匹配')
    if cnt <= 5:
        print('now is b', b)
    hanzi = ''.join(b)

    text = hanzi
    if cnt <= 5:
        print(text)



    phoneme, raw_phoneme = change2phoneme(text, i)
    character_dict.update(phoneme.split('_'))
    
    
    
    with open(output_txt, 'a', encoding='utf-8') as f:
        # f.writelines(phoneme + '\n')
        f.writelines(phoneme + '|1' + '\n')
    with open(output_crystal_txt, 'a', encoding='utf-8') as f:
        f.writelines(raw_phoneme + '\n')
    # break

print('errors:', errors)
print('all is:', character_dict)
print('len:', len(character_dict))





