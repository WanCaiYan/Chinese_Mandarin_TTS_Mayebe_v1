
// CrystalDllSample.cpp : Defines the entry point for the console application.
//

#include <cstring>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "CrystalTTS.h"

const int MAX_SIZE = 300;

int main(int argc, char* argv[])
{
    // Specify the paths where speech library and dictionary data are stored
    std::wstring strDirSpeechLib  = L"";
    //std::wstring strDirDictionary = L"./CrystalTTS/data/putonghua/text";
    std::wstring strDirDictionary = L"../data/putonghua/text";

    // Initialize the TTS engine
    printf("Initializing...");
    handle ttsHandle = ttsInitialize(L"zh-cmn", strDirDictionary.c_str(), strDirSpeechLib.c_str());
    if (ttsHandle == NULL)
    {
        std::printf("\nTTS engine initialization error, please ensure the correct path.\n");
        return -1;
    }
    printf("Done!\n");



    FILE *p_source = NULL;
    char *fname = "tmp_utf_del.txt";
    if(argc >= 2) {
        fname = argv[1];
    }
    wchar_t *outname = L"test.txt";
    if(argc >= 3) {
        outname = (wchar_t *)malloc(1000);
        //char * str = "test.txt";
        swprintf(outname, 1000, L"%hs", argv[2]);
    }
    wchar_t buffer[MAX_SIZE] = { 0 };
    wchar_t *src;

    if((p_source = fopen(fname, "r,ccs=utf-8")))
    {
        while(!feof(p_source))
        {
            src = fgetws(buffer, MAX_SIZE, p_source);
            if (!src) {
                //printf("return NULL\n");
                break;
            } else {
                //printf("%s", src);
                break;
            }
        }
        //printf("\n");
    }




    //std::wstring input;
    //input = L"这是个测试，有没有装你好啊汪仔。";

    // Synthesize the text
    if (!ttsTextAnalysis(ttsHandle, src, true, outname))
    {
        printf("TTS synthesizing error!\n");
    }
    else
    {
        printf("TTS synthesized OK\n");
    }

    // Close the TTS engine
    if (!ttsTerminate(ttsHandle))
    {
        printf("TTS engine close error!\n");
        return -1;
    }
    printf("Engine exits successfully. Bye-bye.\n");

    return 0;
}
