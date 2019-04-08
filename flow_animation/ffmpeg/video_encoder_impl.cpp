#define VIDEO_ENC_EXPORT
#include "video_encoder_impl.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#define RET_NSUC(a) do {if ((a) < 0) return a;} while (0)



int Video_Encoder::write_callback(void *opaque, unsigned char *buf, int buf_size)
{
    Video_Encoder *t = (Video_Encoder*)opaque;
	return t->m_writer->write(buf, buf_size);
}


int64_t Video_Encoder::seek_callback(void *opaque, int64_t offset, int whence)
{
    Video_Encoder *t = (Video_Encoder*)opaque;
	return t->m_writer->seek(offset, whence);
}


Video_Encoder::Video_Encoder()
{
	av_log_set_level(AV_LOG_QUIET);

    m_writer        = NULL;
    m_width			= 0;
	m_height		= 0;
	m_buffer		= NULL;
	m_encode_video	= 1;

    m_avio_ctx      = NULL;
    m_fmt_ctx       = NULL;
    m_video_stream  = NULL;
    m_frame         = NULL;
    m_fmt           = NULL;
    m_video_codec   = NULL;
    m_next_pts      = 0;

	ma_fmt_ctx		= NULL;
	ma_audio_stream = NULL;
	ma_encode_audio = 0;
	ma_next_pts		= 0;

    m_sws_ctx       = NULL;
    m_cust_err_descr= NULL;

	avcodec_register_all();
	av_register_all();
}


Video_Encoder::~Video_Encoder()
{
    avcodec_close(m_video_stream->codec);
    avformat_free_context(m_fmt_ctx);

	avformat_free_context(ma_fmt_ctx);

    av_frame_free(&m_frame);
    av_free(m_avio_ctx);

    sws_freeContext(m_sws_ctx);

	delete[] m_buffer;
}


int Video_Encoder::Init(IVideo_Writer *w, int width, int height, int fps, int bitrate)
{
    int ret_c;

    m_writer  = w;
    m_width   = (width / 2) * 2;
    m_height  = (height / 2) * 2;
	m_sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGR24, width, height, AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
    if (! m_sws_ctx) {
        m_cust_err_descr = "sws_getContext fail";
        return -1;
    }

	const char *filename = "out.mp4";

	m_buffer   = new unsigned char[buf_size];
	m_avio_ctx = avio_alloc_context(m_buffer, buf_size, 1, this, NULL, write_callback, seek_callback);
	m_fmt      = av_guess_format(NULL, filename, NULL);
	m_fmt_ctx  = avformat_alloc_context();
	m_fmt_ctx->pb = m_avio_ctx;
    m_fmt_ctx->oformat = m_fmt;

    ret_c = AddVideoStream(&m_video_codec, AV_CODEC_ID_H264, fps, bitrate);
    RET_NSUC(ret_c);

	if (ma_encode_audio) {
		ret_c = AddAudioStream();
		RET_NSUC(ret_c);
	}

	AVDictionary *opt = NULL;
    ret_c = OpenVideo(m_video_codec, opt);
    RET_NSUC(ret_c);

//	av_dump_format(m_fmt_ctx, 0, filename, 1);

    return  avformat_write_header(m_fmt_ctx, &opt);
}


int Video_Encoder::WriteVideoFrame(unsigned char *rgb_data, int pitch, int duration) 
{
	av_frame_make_writable(m_frame);

	unsigned char *inData[1] = { rgb_data };
	int inLinesiaze[1] = { pitch };
	sws_scale(m_sws_ctx, inData, inLinesiaze, 0, m_height, m_frame->data, m_frame->linesize);

	m_frame->pts = m_next_pts;
	m_next_pts += duration;

	int got_packet;
	return EncodeAndWriteframe(m_frame, got_packet);
}


int Video_Encoder::WriteAudioFrame(int duration)
{
	int ret_c;
	AVPacket pkt;
	memset(&pkt, 0, sizeof(pkt));
	av_init_packet(&pkt);

	ret_c = av_read_frame(ma_fmt_ctx, &pkt);
	RET_NSUC(ret_c);

	if (pkt.stream_index != ma_stream_idx) {
		av_packet_unref(&pkt);
		return 0;
	}

	pkt.stream_index = ma_audio_stream->id;

	pkt.pts = av_rescale_q_rnd(pkt.pts, ma_fmt_ctx->streams[ma_stream_idx]->time_base, ma_audio_stream->time_base, AV_ROUND_PASS_MINMAX/*|AV_ROUND_NEAR_INF*/);
	pkt.dts = av_rescale_q_rnd(pkt.dts, ma_fmt_ctx->streams[ma_stream_idx]->time_base,  ma_audio_stream->time_base, AV_ROUND_PASS_MINMAX/*|AV_ROUND_NEAR_INF*/);
	pkt.duration = av_rescale_q(pkt.duration, ma_fmt_ctx->streams[ma_stream_idx]->time_base, ma_audio_stream->time_base);
	pkt.pos = -1;

	ret_c = av_interleaved_write_frame(m_fmt_ctx, &pkt);
	RET_NSUC(ret_c);

	av_packet_unref(&pkt);

	ma_next_pts += duration;

	return ret_c;
}


int Video_Encoder::Addframe(unsigned char *rgb_data, int pitch, int duration)
{
	while (m_encode_video) {
		if (ma_encode_audio && av_compare_ts(m_next_pts, m_video_stream->time_base, ma_next_pts, ma_audio_stream->time_base) > 0) {
			ma_encode_audio = !WriteAudioFrame(duration);
			//ma_next_pts++;
		}
		else {
			m_encode_video = !WriteVideoFrame(rgb_data, pitch, duration);
			break;
		}
	}

	return m_encode_video;
}


int Video_Encoder::Finalize()
{
    int ret_c;
    int got_packet;

    do {
        ret_c = EncodeAndWriteframe(NULL, got_packet);
        RET_NSUC(ret_c);

    } while (got_packet);

    return av_write_trailer(m_fmt_ctx);
}


void Video_Encoder::Destroy()
{
    delete this;
}


void Video_Encoder::GetErrorDescription(char *buf, size_t len, int code)
{
    if (m_cust_err_descr) {
        strncpy(buf, m_cust_err_descr, len);
        buf[len-1] = 0;
    }
    else {
        av_make_error_string(buf, len, code);
    }
}


int Video_Encoder::EncodeAndWriteframe(AVFrame *frame, int &got_packet)
{
    AVPacket pkt;
    memset(&pkt, 0, sizeof(pkt));
	av_init_packet(&pkt);

    got_packet = 0;
    int ret_c = avcodec_encode_video2(m_video_stream->codec, &pkt, frame, &got_packet);
    RET_NSUC(ret_c);

	if (got_packet) {
		/* rescale output packet timestamp values from codec to stream timebase */
		av_packet_rescale_ts(&pkt, m_video_stream->codec->time_base, m_video_stream->time_base);
		pkt.stream_index = m_video_stream->index;

        ret_c = av_interleaved_write_frame(m_fmt_ctx, &pkt);
        RET_NSUC(ret_c);
	}
    return 0;
}


int Video_Encoder::AddAudioStream()
{
	int ret_c;

	ma_audio_stream = avformat_new_stream(m_fmt_ctx, NULL);
	if (!ma_audio_stream) {
		m_cust_err_descr = "Could not allocate stream";
		return -1;
	}

	ma_audio_stream->id = m_fmt_ctx->nb_streams - 1;

	ret_c = avcodec_copy_context(ma_audio_stream->codec, ma_fmt_ctx->streams[ma_stream_idx]->codec);
	RET_NSUC(ret_c);

	ma_audio_stream->codec->codec_tag = 0;

	/// Some formats want stream headers to be separate.
	if (m_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
		ma_audio_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
	}

	return 0;
}


int Video_Encoder::AddVideoStream(AVCodec **codec, enum AVCodecID codec_id, int fps, int bitrate)
{
    *codec = avcodec_find_encoder(codec_id);
    if (!*codec) {
        m_cust_err_descr = "Could not find encoder";
        return -1;
    }
    m_video_stream = avformat_new_stream(m_fmt_ctx, *codec);
    if (!m_video_stream) {
        m_cust_err_descr = "Could not allocate stream";
        return -1;
    }

    m_video_stream->id = m_fmt_ctx->nb_streams - 1;
    AVCodecContext *&c = m_video_stream->codec;
	c = avcodec_alloc_context3(*codec);

    c->codec_id = codec_id;
    c->bit_rate = bitrate;
    /* Resolution must be a multiple of two. */
    c->width    = m_width;
    c->height   = m_height;
    m_video_stream->time_base.num = 1;
    m_video_stream->time_base.den = fps;
    c->time_base       = m_video_stream->time_base;
    c->gop_size      = 10; /* emit one intra frame every twelve frames at most */
    c->pix_fmt       = AV_PIX_FMT_YUV420P;
    int ret_c = av_opt_set(c->priv_data, "profile", "baseline",  AV_OPT_SEARCH_CHILDREN);
    RET_NSUC(ret_c);

    /* Some formats want stream headers to be separate. */
    if (m_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }



    return 0;
}


int Video_Encoder::OpenVideo(AVCodec *codec, AVDictionary *opt_arg)
{
    int ret_c;

	AVCodecContext *c = m_video_stream->codec;
	AVDictionary *opt = NULL;
	av_dict_copy(&opt, opt_arg, 0);
    ret_c = avcodec_open2(c, codec, &opt);
	av_dict_free(&opt);

    RET_NSUC(ret_c);

	m_frame = av_frame_alloc();
	if (! m_frame) {
        m_cust_err_descr = "av_frame_alloc fail";
        return -1;
	}
	m_frame->width  = c->width;
	m_frame->height = c->height;
	m_frame->format = c->pix_fmt;
	if (av_frame_get_buffer(m_frame, 32) < 0) {
        m_cust_err_descr = "av_frame_get_buffer fail";
        return -1;
	}
    return 0;
}


int	Video_Encoder::AddAudio(std::string filename)
{
	int ret_c;

	ret_c = avformat_open_input(&ma_fmt_ctx, filename.c_str(), 0, 0);
	RET_NSUC(ret_c);

	ret_c = avformat_find_stream_info(ma_fmt_ctx, 0);
	RET_NSUC(ret_c);

	ma_stream_idx = -1;
	for (size_t i = 0; i < ma_fmt_ctx->nb_streams; i++) {
		if (ma_fmt_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			ma_stream_idx = i;
			break;
		}
	}
	RET_NSUC(ma_stream_idx);

	ma_encode_audio = 1;

	return 0;
}


void video_encoder_register()
{
    avcodec_register_all();
    av_register_all();
}


IVideo_Encoder *video_encoder_create()
{
    return new Video_Encoder();
}
