/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVPIPE_H
#define NVPIPE_H

#include <stdlib.h>
#include <stdint.h>

#define NVPIPE_WITH_ENCODER
#define NVPIPE_WITH_OPENGL

#ifdef _WIN32
#	define NVPIPE_EXPORT __declspec(dllexport)
#else
#	define NVPIPE_EXPORT __attribute__((visibility("default")))
#endif

extern "C"
{

typedef void NvPipe;


/**
 * Available video codecs in NvPipe.
 */
typedef enum {
    NVPIPE_H264,
    NVPIPE_HEVC
} NvPipe_Codec;


/**
 * Compression type used for encoding. Lossless produces larger output.
 */
typedef enum {
    NVPIPE_LOSSY,
    NVPIPE_LOSSLESS
} NvPipe_Compression;


/**
 * Format of the input frame.
 */
typedef enum {
    NVPIPE_RGBA32,
    NVPIPE_UINT4,
    NVPIPE_UINT8,
    NVPIPE_UINT16,
    NVPIPE_UINT32
} NvPipe_Format;


#ifdef NVPIPE_WITH_ENCODER

/**
 * @brief Creates a new encoder instance.
 * @param format Format of input frame.
 * @param codec Possible codecs are H.264 and HEVC if available.
 * @param compression Lossy or lossless compression.
 * @param bitrate Bitrate in bit per second, e.g., 32 * 1000 * 1000 = 32 Mbps (for lossy compression only).
 * @param targetFrameRate At this frame rate the effective data rate approximately equals the bitrate (for lossy compression only).
 * @param width Initial width of the encoder.
 * @param height Initial height of the encoder.
 * @return NULL on error.
 */
NVPIPE_EXPORT NvPipe* NvPipe_CreateEncoder(NvPipe_Format format, NvPipe_Codec codec, NvPipe_Compression compression, uint64_t bitrate, uint32_t targetFrameRate, uint32_t width, uint32_t height);


/**
 * @brief Reconfigures the encoder with a new bitrate and target frame rate.
 * @param nvp Encoder instance.
 * @param bitrate Bitrate in bit per second, e.g., 32 * 1000 * 1000 = 32 Mbps (for lossy compression only).
 * @param targetFrameRate At this frame rate the effective data rate approximately equals the bitrate (for lossy compression only).
 */
NVPIPE_EXPORT void NvPipe_SetBitrate(NvPipe* nvp, uint64_t bitrate, uint32_t targetFrameRate);


/**
 * @brief Encodes a single frame from device or host memory.
 * @param nvp Encoder instance.
 * @param src Device or host memory pointer.
 * @param srcPitch Pitch of source memory.
 * @param dst Host memory pointer for compressed output.
 * @param dstSize Available space for compressed output.
 * @param width Width of input frame in pixels.
 * @param height Height of input frame in pixels.
 * @param forceIFrame Enforces an I-frame instead of a P-frame.
 * @return Size of encoded data in bytes or 0 on error.
 */
NVPIPE_EXPORT uint64_t NvPipe_Encode(NvPipe* nvp, const void* src, uint64_t srcPitch, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame);


#ifdef NVPIPE_WITH_OPENGL

/**
 * @brief encodeTexture Encodes a single frame from an OpenGL texture.
 * @param nvp Encoder instance.
 * @param texture OpenGL texture ID.
 * @param target OpenGL texture target.
 * @param dst Host memory pointer for compressed output.
 * @param dstSize Available space for compressed output. Will be overridden by effective compressed output size.
 * @param width Width of frame in pixels.
 * @param height Height of frame in pixels.
 * @param forceIFrame Enforces an I-frame instead of a P-frame.
 * @return Size of encoded data in bytes or 0 on error.
 */
NVPIPE_EXPORT uint64_t NvPipe_EncodeTexture(NvPipe* nvp, uint32_t texture, uint32_t target, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame);


/**
 * @brief encodePBO Encodes a single frame from an OpenGL pixel buffer object (PBO).
 * @param nvp Encoder instance.
 * @param pbo OpenGL PBO ID.
 * @param dst Host memory pointer for compressed output.
 * @param dstSize Available space for compressed output. Will be overridden by effective compressed output size.
 * @param width Width of frame in pixels.
 * @param height Height of frame in pixels.
  * @param forceIFrame Enforces an I-frame instead of a P-frame.
 * @return Size of encoded data in bytes or 0 on error.
 */
NVPIPE_EXPORT uint64_t NvPipe_EncodePBO(NvPipe* nvp, uint32_t pbo, uint8_t* dst, uint64_t dstSize, uint32_t width, uint32_t height, bool forceIFrame);

#endif

#endif

#ifdef NVPIPE_WITH_DECODER

/**
 * @brief Creates a new decoder instance.
 * @param format Format of output frame.
 * @param codec Possible codecs are H.264 and HEVC if available.
 * @param width Initial width of the decoder.
 * @param height Initial height of the decoder.
 * @return NULL on error.
 */
NVPIPE_EXPORT NvPipe* NvPipe_CreateDecoder(NvPipe_Format format, NvPipe_Codec codec, uint32_t width, uint32_t height);


/**
 * @brief Decodes a single frame to device or host memory.
 * @param nvp Decoder instance.
 * @param src Compressed frame data in host memory.
 * @param srcSize Size of compressed data.
 * @param dst Device or host memory pointer.
 * @param width Width of frame in pixels.
 * @param height Height of frame in pixels.
 * @return Size of decoded data in bytes or 0 on error.
 */
NVPIPE_EXPORT uint64_t NvPipe_Decode(NvPipe* nvp, const uint8_t* src, uint64_t srcSize, void* dst, uint32_t width, uint32_t height);


#ifdef NVPIPE_WITH_OPENGL

/**
 * @brief Decodes a single frame to an OpenGL texture.
 * @param nvp Decoder instance.
 * @param src Compressed frame data in host memory.
 * @param srcSize Size of compressed data.
 * @param texture OpenGL texture ID.
 * @param target OpenGL texture target.
 * @param width Width of frame in pixels.
 * @param height Height of frame in pixels.
 * @return Size of decoded data in bytes or 0 on error.
 */
NVPIPE_EXPORT uint64_t NvPipe_DecodeTexture(NvPipe* nvp, const uint8_t* src, uint64_t srcSize, uint32_t texture, uint32_t target, uint32_t width, uint32_t height);


/**
 * @brief Decodes a single frame to an OpenGL pixel buffer object (PBO).
 * @param nvp Decoder instance.
 * @param src Compressed frame data in host memory.
 * @param srcSize Size of compressed data.
 * @param pbo OpenGL PBO ID.
 * @param width Width of frame in pixels.
 * @param height Height of frame in pixels.
 * @return Size of decoded data in bytes or 0 on error.
 */
NVPIPE_EXPORT uint64_t NvPipe_DecodePBO(NvPipe* nvp, const uint8_t* src, uint64_t srcSize, uint32_t pbo, uint32_t width, uint32_t height);

#endif

#endif


/**
 * @brief Cleans up an encoder or decoder instance.
 * @param nvp The encoder or decoder instance to destroy.
 */
NVPIPE_EXPORT void NvPipe_Destroy(NvPipe* nvp);


/**
 * @brief Returns an error message for the last error that occured.
 * @param nvp Encoder or decoder. Use NULL to get error message if encoder or decoder creation failed.
 * @return Returned string must not be deleted.
 */
NVPIPE_EXPORT const char* NvPipe_GetError(NvPipe* nvp);

}

#endif

