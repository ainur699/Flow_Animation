#pragma once

#include "video_encoder.h"

#define __STDC_FORMAT_MACROS
#define __STDC_CONSTANT_MACROS

extern "C"
{
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}



class Video_Encoder : public IVideo_Encoder {
public:
    Video_Encoder();
    ~Video_Encoder();

    virtual int  Init(IVideo_Writer *, int width, int height, int fps, int bitrate);
    virtual int  Addframe(unsigned char *rgb_data, int pitch, int duration);
    virtual int  Finalize();
    virtual void GetErrorDescription(char *buf, size_t len, int code);
    virtual void Destroy();
	virtual int	 AddAudio(std::string filename);

private:
    IVideo_Writer   *m_writer;
	int              m_width;
	int              m_height;
	unsigned char   *m_buffer;
	SwsContext      *m_sws_ctx;

	AVFormatContext *ma_fmt_ctx;
	int				ma_stream_idx;
	AVStream        *ma_audio_stream;
	int				ma_encode_audio;
	int64_t         ma_next_pts;

	AVIOContext     *m_avio_ctx;
    AVFormatContext *m_fmt_ctx;
    AVStream        *m_video_stream;
    AVFrame         *m_frame;
    AVOutputFormat  *m_fmt;
	AVCodec         *m_video_codec;
    int64_t          m_next_pts;
    const char      *m_cust_err_descr;
	int				m_encode_video;

	enum {buf_size = 0x400 * 10};

	int WriteVideoFrame(unsigned char *rgb_data, int pitch, int duration);
	int WriteAudioFrame(int duration);
    int AddVideoStream(AVCodec **codec, enum AVCodecID codec_id, int fps, int bitrate);
    int OpenVideo(AVCodec *codec, AVDictionary *opt_arg);
    int EncodeAndWriteframe(AVFrame *, int &got_packet);

	int AddAudioStream();

	static int write_callback(void *opaque, unsigned char *buf, int buf_size);
	static int64_t seek_callback(void *opaque, int64_t offset, int whence);
};
