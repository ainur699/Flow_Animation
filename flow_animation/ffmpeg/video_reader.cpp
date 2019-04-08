#include "video_reader.h"
#include "opencv2/imgproc.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#define CHECK_RET(a) if ((a) < 0) return a;



int FgVideoCapture::ReadFromFile(std::string video_fname)
{
	frame_buf.clear();
	m_position = 0;
	m_fps = 0;

	AVFormatContext   *pFormatCtx = NULL;
	AVCodecContext    *pCodecCtx = NULL;
	AVCodec           *pCodec = NULL;
	AVFrame           *pFrame = NULL;
	AVFrame           *pFrameRGB = NULL;
	uint8_t           *buffer = NULL;
	SwsContext		  *sws_ctx = NULL;
	AVPacket          packet;
	int				  ret;
	int               frameFinished;

	/// Register all formats and codecs
	av_register_all();

	/// Open video file
	ret = avformat_open_input(&pFormatCtx, video_fname.c_str(), NULL, NULL);
	CHECK_RET(ret);

	/// Retrieve stream information
	ret = avformat_find_stream_info(pFormatCtx, NULL);
	CHECK_RET(ret);

	/// Find the first video stream
	int videoStream = -1;
	for (size_t i = 0; i<pFormatCtx->nb_streams; i++)
		if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			videoStream = i;
			break;
		}
	if (videoStream == -1) return -1;

	/// Get a pointer to the codec context for the video stream
	pCodecCtx = pFormatCtx->streams[videoStream]->codec;
	m_fps = pCodecCtx->time_base.den;

	/// Find the decoder for the video stream
	pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
	if (pCodec == NULL) return -1;

	/// Open codec
	ret = avcodec_open2(pCodecCtx, pCodec, NULL);
	CHECK_RET(ret);

	/// Allocate video frame
	pFrame = av_frame_alloc();
	if (pFrame == NULL) return -1;

	/// Allocate an AVFrame structure
	pFrameRGB = av_frame_alloc();
	if (pFrameRGB == NULL) return -1;

	/// Determine required buffer size and allocate buffer
	int numBytes = avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
	buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));

	/// Assign appropriate parts of buffer to image planes in pFrameRGB
	/// pFrameRGB is an AVFrame, but AVFrame is a superset of AVPicture
	avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);

	/// initialize SWS context for software scaling
	sws_ctx = sws_getContext(
		pCodecCtx->width,
		pCodecCtx->height,
		pCodecCtx->pix_fmt,
		pCodecCtx->width,
		pCodecCtx->height,
		AV_PIX_FMT_RGB24,
		SWS_BILINEAR,
		NULL,
		NULL,
		NULL);

	int i = 0;
	while (av_read_frame(pFormatCtx, &packet) >= 0) {
		/// Is this a packet from the video stream?
		if (packet.stream_index == videoStream) {
			/// Decode video frame
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

			/// Did we get a video frame
			if (frameFinished) {
				/// Convert the image from its native format to RGB
				sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
					pFrame->linesize, 0, pCodecCtx->height,
					pFrameRGB->data, pFrameRGB->linesize);

				///Save the frame
				cv::Mat &&cv_img = cv::Mat(pCodecCtx->height, pCodecCtx->width, CV_8UC3, pFrameRGB->data[0], pFrameRGB->linesize[0]);
				cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);

				double max = cv_img.rows > cv_img.cols ? cv_img.rows : cv_img.cols;
				double ratio = 400.0 / max;
				cv::resize(cv_img, cv_img, cv::Size(), ratio, ratio);

				frame_buf.push_back(cv_img);

				/// Stop reading frames
				if (frame_buf.size() > 350) break;
			}
		}

		/// Free the packet that was allocated by av_read_frame
		av_free_packet(&packet);
	}

	/// clean all
	av_free(buffer);
	av_frame_free(&pFrameRGB);
	av_frame_free(&pFrame);
	avcodec_close(pCodecCtx);
	avformat_close_input(&pFormatCtx);

	return 0;
}


cv::Mat FgVideoCapture::getFrame()
{
	return frame_buf[m_position++];
}


int FgVideoCapture::size() const
{
	return frame_buf.size();
}


void FgVideoCapture::setPos(uint pos)
{
	if (pos >= 0 && pos < frame_buf.size()) {
		m_position = pos;
	}
}
