
import ctypes
from ctypes import *
import os



class CrystalTTS(object):
    #def __init__(self, dllpath = u'CrystalTTS/demo/CrystalDll.so'):
    def __init__(self, dllpath = u'./CrystalDll.so'):
        self._dllpath = dllpath
        print(os.path.abspath(dllpath))
        self._libc = ctypes.cdll.LoadLibrary(self._dllpath)
        print('load ok')
        self._tts = None

    #def initialize(self, textpath = u'CrystalTTS/data/putonghua/text'):
    def initialize(self, textpath = u'../data/putonghua/text'):
        langtag = c_wchar_p(u"zh-cmn")
        dirtext = c_wchar_p(textpath)
        dirvoice= c_wchar_p(u"")
        
        class handle_t(Structure):
            pass
        self._libc.ttsInitialize.restype = POINTER(handle_t)

        self._tts = self._libc.ttsInitialize(langtag, dirtext, dirvoice)

    def terminate(self):
         self._libc.ttsTerminate(self._tts)

    def textAnalyze(self, text, filename):
        print('first', text)
        newtext = c_wchar_p(text)
        newfile = c_wchar_p(filename)
        print('show', type(self._tts), self._tts)
        print('c_wchar_p', newtext)
        tmp = self._libc.ttsTextAnalysis(self._tts, newtext, c_int(1), newfile)
        print('no here', tmp)
        print("ok?????")
        return tmp


