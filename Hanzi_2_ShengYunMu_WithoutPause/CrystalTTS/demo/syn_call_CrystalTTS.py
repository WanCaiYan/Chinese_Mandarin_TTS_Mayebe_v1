from CrystalTTS import CrystalTTS

c2p = CrystalTTS()
c2p.initialize()
s = '我和我的祖国，一刻也不能分割。'
print(s)
file_name = 'wohe.txt'
c2p.textAnalyze(s, file_name)