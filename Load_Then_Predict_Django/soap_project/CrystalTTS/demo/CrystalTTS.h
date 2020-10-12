//
//  Crystal Text-to-Speech Engine
//
//  Copyright (c) 2007 THU-CUHK Joint Research Center for
//  Media Sciences, Technologies and Systems. All rights reserved.
//
//  http://mjrc.sz.tsinghua.edu.cn
//
//  Redistribution and use in source and binary forms, with or without
//  modification, is not allowed, unless a valid written license is
//  granted by THU-CUHK Joint Research Center.
//
//  THU-CUHK Joint Research Center has the rights to create, modify,
//  copy, compile, remove, rename, explain and deliver the source codes.
//

#ifndef _CST_TTS_BASE_CMN_TYPE_H_
#define _CST_TTS_BASE_CMN_TYPE_H_

typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;
typedef unsigned int        icode_t;
typedef unsigned long   ulong;
typedef unsigned short  ushort;
typedef unsigned int    uint;
typedef unsigned char   byte;
//typedef unsigned short  icode_t;
typedef void*           handle;
#define  NULL            0

#endif//_CST_TTS_BASE_CMN_TYPE_H_



#ifndef _CST_TTS_CRYSTAL_DLL_H_
#define _CST_TTS_CRYSTAL_DLL_H_

#   define CST_EXPORT extern

#ifdef __cplusplus
extern "C" {
#endif

///
/// @brief  Initialize the TTS engine
///
/// @param  [in] szLangTag      The language tag. "zh-cmn" for Chinese Putonghua (Mandarin), "zh-yue" for Chinese Cantonese
/// @param  [in] szDictPath     The path where the dictionary data are stored
/// @param  [in] szSplibPath    The path where the speech library data are stored
///
/// @return The handle of the initialized TTS engine, or NULL if failed.
///
CST_EXPORT handle ttsInitialize(const wchar_t* szLangTag, const wchar_t* szDictPath, const wchar_t* szSplibPath);

///
/// @brief  Terminate the TTS engine, and close all data
///
/// @param  [in] ttsHandle      The handle of the TTS engine to be closed
///
/// @return Whether operation is successful
///
CST_EXPORT bool ttsTerminate(handle ttsHandle);

///
/// @brief  Convert the text to phoneme list
///
/// @param  [in]  ttsHandle      The handle of the TTS engine
/// @param  [in]  szText         The text to be synthesized
/// @param  [in]  bPartialSSML   The input text contains partial SSML tag
/// @param  [in]  szOutFileName  The file name to write the pinyin (initial-final) string result
///
/// @return Whether operation is successful
///
CST_EXPORT bool ttsTextAnalysis(handle ttsHandle, const wchar_t* szText, bool bPartialSSML, const wchar_t* szOutFileName);

#ifdef __cplusplus
}
#endif


#endif//_CST_TTS_CRYSTAL_DLL_H_
