#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include "ffmpeg/video_encoder.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include "render/GLSLProgram.h"
#include "render/Texture.hpp"
#include <thread>
#include <filesystem>

///dlib
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>


#define __DEBUG
#define __GPU
//#define __CPU

#if defined( _WIN32 )
	#ifdef __DEBUG
		#define ASSERT(x) if(!(x)) __debugbreak()
	#else
		#define ASSERT(x)
	#endif
#endif

enum class Material
{
	WATER, HAIR, SKY, TREE
};


template<class Type>
void BilinInterp(const cv::Mat &I, double x, double y, Type *dst)
{
	int x1 = (int)std::floor(x);
	int y1 = (int)std::floor(y);
	int x2 = (int)std::ceil(x);
	int y2 = (int)std::ceil(y);
	if (x1 < 0 || x2 >= I.cols || y1 < 0 || y2 >= I.rows) {
		for (int i = 0; i < I.channels(); i++)
		{
			*dst++ = (Type)0.0;
		}
		return;
	};

	const Type *p1 = (Type*)I.ptr(y1, x1);
	const Type *p2 = (Type*)I.ptr(y1, x2);
	const Type *p3 = (Type*)I.ptr(y2, x1);
	const Type *p4 = (Type*)I.ptr(y2, x2);
	for (int i = 0; i < I.channels(); i++)
	{
		float c1 = p1[i] + ((float)p2[i] - p1[i]) * (x - x1);
		float c2 = p3[i] + ((float)p4[i] - p3[i]) * (x - x1);
		*dst++ = (Type)(c1 + (c2 - c1) * (y - y1));
	}
}

struct MemoryStruct {
	char *memory;
	size_t size;
};

size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp)
{
	size_t realsize = size * nmemb;
	MemoryStruct *mem = reinterpret_cast<MemoryStruct*>(userp);

	auto *ptr = std::realloc(mem->memory, mem->size + realsize + 1);
	if (!ptr) {
		return 0;
	}

	mem->memory = reinterpret_cast<char*>(ptr);
	std::memcpy(&(mem->memory[mem->size]), contents, realsize);
	mem->size += realsize;
	mem->memory[mem->size] = 0;

	return realsize;
}

size_t read_callback(void *dest, size_t size, size_t nmemb, void *userp)
{
	MemoryStruct *mem = reinterpret_cast<MemoryStruct *>(userp);
	size_t buffer_size = size * nmemb;

	if (mem->size) {
		size_t copy_this_much = (mem->size > buffer_size) ? buffer_size : mem->size;

		memcpy(dest, mem->memory, copy_this_much);

		mem->memory += copy_this_much;
		mem->size -= copy_this_much;

		return copy_this_much;
	}

	return 0;
}

int GetMask(const cv::Mat &src, cv::Mat &dst, Material material)
{
	CURLcode res;

	struct MemoryStruct result_chunk;
	result_chunk.memory = reinterpret_cast<char*>(malloc(1));
	result_chunk.size = 0;

	curl_global_init(CURL_GLOBAL_ALL);
	CURL *curl = curl_easy_init();

	curl_httppost *formpost = NULL;
	curl_httppost *lastptr = NULL;

	std::vector<uchar> img_data;
	cv::imencode(".jpg", src, img_data);

	if (material == Material::HAIR)
	{
		curl_formadd(&formpost,
			&lastptr,
			CURLFORM_COPYNAME, "detector",
			CURLFORM_COPYCONTENTS, "hair",
			CURLFORM_END
		);
	}

	curl_formadd(&formpost,
		&lastptr,
		CURLFORM_COPYNAME, "image",
		CURLFORM_BUFFER, "filename.jpg",
		CURLFORM_BUFFERPTR, img_data.data(),
		CURLFORM_BUFFERLENGTH, img_data.size(),
		CURLFORM_END
	);

#if 0
	curl_formadd(&formpost,
		&lastptr,
		CURLFORM_COPYNAME, "image",
		CURLFORM_FILE, "C:/Users/Ainur/Desktop/image7.jpg",
		CURLFORM_END);

	curl_formadd(&formpost,
		&lastptr,
		CURLFORM_COPYNAME, "image",
		CURLFORM_PTRCONTENTS, img_data.data(),
		CURLFORM_CONTENTSLENGTH, img_data.size(),
		CURLFORM_END);
#endif

	if (curl) {
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

		switch (material) {
		case Material::HAIR: {
			curl_easy_setopt(curl, CURLOPT_URL, "http://admin:show-me-viewer@exp.ws.pho.to/viewer/detect-blob");
			break;
		}
		case Material::WATER: {
			curl_easy_setopt(curl, CURLOPT_URL, "http://67.228.246.51:8056/forward");
			break;
		}
		}

		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result_chunk);

		res = curl_easy_perform(curl);

		if (res == CURLE_OK) {
			std::vector<uchar> buf(result_chunk.memory, result_chunk.memory + result_chunk.size);
			dst = cv::imdecode(buf, cv::ImreadModes::IMREAD_UNCHANGED);

			if (material == Material::WATER) {
				for (int i = 0; i < dst.rows; i++) {
					uchar *p_im = dst.ptr<uchar>(i);
					for (int j = 0; j < dst.cols; j++) {
						*p_im = (*p_im == 5) ? 255 : 0;
						p_im++;
					}
				}
			}

			if (dst.empty()) {
				std::cout << std::string(buf.begin(), buf.end()) << std::endl;
				return -1;
			}
		}

		curl_easy_cleanup(curl);
		curl_formfree(formpost);

		free(result_chunk.memory);
		curl_global_cleanup();
	}

	return res;
}

bool ContourOrientationCW(const std::vector<cv::Point>& contour) {
	if (contour.size() >= 3) {
		cv::Point rm;
		size_t rmIdx;

		for (size_t i = 0; i < contour.size(); i++) {
			const cv::Point& p = contour[i];

			if (p.x > rm.x || p.x == rm.x && p.y > rm.y) {
				rm = p;
				rmIdx = i;
			}
		}

		size_t i = rmIdx - 1; if (i < 0) i = contour.size() - 1;
		const cv::Point& pred = contour[i];
		i = rmIdx + 1; if (i == contour.size()) i = 0;
		const cv::Point& succ = contour[i];

		cv::Vec2i a = pred - rm, b = succ - rm;
		return a[0] * b[1] <= a[1] * b[0];
	}
	return true;
}

void CreateNormalMask(cv::Mat& img, cv::Subdiv2D& subdiv, std::vector<cv::Point2f> &contour, std::vector<cv::Point2f> &normals)
{
	std::vector<std::vector<cv::Point2f>> facets;
	std::vector<cv::Point2f> centers;
	std::vector<cv::Point> ifacet;
	subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
	

	for (size_t i = 0; i < facets.size(); i++)
	{
		ifacet.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++) {
			ifacet[j] = facets[i][j];
		}

		int min_ind = -1;
		float min_val = std::max(img.cols, img.rows);
		
		for (int j = 0; j < contour.size(); j++) {
			if (min_val > cv::norm(contour[j] - centers[i])) {
				min_val = cv::norm(contour[j] - centers[i]);
				min_ind = j;
			}
		}

		fillConvexPoly(img, ifacet, cv::Scalar(normals[min_ind].x, normals[min_ind].y), 8, 0);
	}

	cv::blur(img, img, cv::Size(3, 3));
}

cv::Mat _rendertestmask(cv::Mat& testimage)
{
	cv::Mat rendertestimage;
	std::vector<cv::Mat> channels;
	cv::split(testimage, channels);
	channels.push_back(cv::Mat::zeros(channels[0].rows, channels[0].cols, channels[0].type()));
	cv::merge(channels, rendertestimage);
	return rendertestimage;
}

void CreateVectorField(cv::Mat &mask, cv::Mat &opacity_map, cv::Mat &dst, cv::Point2f dir, std::vector<cv::Point2f> &contour_points, std::vector<cv::Point2f> &normals_points, Material material)
{
	cv::Mat distance(mask.size(), CV_32FC1);
	cv::distanceTransform(mask, distance, cv::DistanceTypes::DIST_L2, cv::DIST_MASK_PRECISE);

	dst.create(mask.size(), CV_32FC2);
	dst.setTo(cv::Scalar::all(0));
	opacity_map.create(mask.size(), CV_32FC1);
	opacity_map.setTo(cv::Scalar::all(1));

	float offset_koeff = 0.02f;
	float offset = offset_koeff * std::min(mask.rows, mask.cols);

	switch (material){
	case Material::HAIR : {
		std::vector<std::vector<cv::Point>> contours;
		std::vector<std::vector<cv::Point2f>> normals;
		cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
		normals.resize(contours.size());
			
		cv::Subdiv2D subdiv(cv::Rect(0, 0, mask.cols, mask.rows));

		for (size_t i = 0; i < contours.size(); i++) {
			cv::approxPolyDP(contours[i], contours[i], 3, true);
			if (contours[i].size() < 3) contours[i].clear();

			if (ContourOrientationCW(contours[i])) {
				std::reverse(contours[i].begin(), contours[i].end());
			}

			normals[i].resize(contours[i].size());

			for (size_t j = 0; j < contours[i].size(); j++) {
				int pred_ind = j - 1; if (pred_ind < 0) pred_ind = contours[i].size() - 1;
				int succ_ind = (j + 1) % contours[i].size();
				const cv::Point &pt = contours[i][j];
				const cv::Point &pt_pred = contours[i][pred_ind];
				const cv::Point &pt_succ = contours[i][succ_ind];
				
				cv::Point2f a = pt - pt_pred;
				cv::Point2f b = pt_succ - pt;

				cv::Point2f norm1 = cv::Point2f(-a.y, a.x) / std::hypot(a.y, a.x);
				cv::Point2f norm2 = cv::Point2f(-b.y, b.x) / std::hypot(b.y, b.x);
				normals[i][j] = 0.5f * (norm1 + norm2);
				subdiv.insert(contours[i][j]);
			}
		}

		
		contour_points.clear();
		normals_points.clear();
		for (int ii = 0; ii < contours.size(); ii++) {
			for (int jj = 0; jj < contours[ii].size(); jj++) {
				contour_points.push_back(cv::Point2f(contours[ii][jj]));
				normals_points.push_back(cv::Point2f(normals[ii][jj]));
			}
		}
		cv::Mat normal_mask(mask.size(), CV_32FC2, cv::Scalar::all(0));
		CreateNormalMask(normal_mask, subdiv, contour_points, normals_points);
		cv::Point2f dir_norm = dir / cv::norm(dir);

		for (int i = 0; i < dst.rows; i++) {
			cv::Vec2f *p_dst	= dst.ptr<cv::Vec2f>(i);
			cv::Vec2f *p_norm	= normal_mask.ptr<cv::Vec2f>(i);
			float *p_dist		= distance.ptr<float>(i);
			float *p_op			= opacity_map.ptr<float>(i);

			for (int j = 0; j < dst.cols; j++)
			{
#if 1
				if (p_dist[j] >= offset) {
					p_dst[j] = dir;
					p_op[j] = 1;
				}
				else {
					float d = dir_norm.x * p_norm[j][0] + dir_norm.y * p_norm[j][1];
					float p = 0.45f * d + 0.55f;
					float x = p_dist[j] / offset;
					float alpha = std::pow(x, p);
					p_dst[j] = alpha * dir;

					if (p_dist[j] == 0) {
						p_op[j] = 1;
					}
					else {
						if (d >= 0) {
							p_op[j] = 1.f;
						}
						else {
							float p2 = -0.45f * d + 0.55f;
							float alpha2 = std::pow(x, p2);
							p_op[j] = alpha2;
						}
					}
				}

				int m_mal = std::min(std::min(i, j), std::min(dst.rows - i, dst.cols - j));
				if (m_mal < offset) {
					p_op[j] *= m_mal / offset;
				}
#else
				p_dst[j] = dir * std::min(p_dist[j] / offset, 1.f);
#endif
			}
		}

		break;
	}
	case Material::WATER: {
		cv::line(mask, cv::Point(0, 0), cv::Point(mask.cols - 1, 0), cv::Scalar::all(0));
		cv::line(mask, cv::Point(mask.cols - 1, mask.rows - 1), cv::Point(mask.cols - 1, 0), cv::Scalar::all(0));
		cv::line(mask, cv::Point(mask.cols - 1, mask.rows - 1), cv::Point(0, mask.rows - 1), cv::Scalar::all(0));
		cv::line(mask, cv::Point(0, 0), cv::Point(0, mask.rows - 1), cv::Scalar::all(0));


		for (int i = 0; i < dst.rows; i++) {
			cv::Vec2f *p_dst	= dst.ptr<cv::Vec2f>(i);
			float *p_dist		= distance.ptr<float>(i);

			for (int j = 0; j < dst.cols; j++)
			{
				p_dst[j] = dir * std::min(p_dist[j] / offset, 1.f);
			}
		}
		break;
	}
	}

#if 1
	cv::Mat vec[2];
	cv::split(dst, vec);
	cv::magnitude(vec[0], vec[1], vec[0]);
	cv::Mat_<uchar> show(200 / cv::norm(dir) * vec[0]);
	cv::imwrite("res_debug/vector_field.png", show);

	cv::Mat_<uchar> opacity_mat(255 * opacity_map);
	cv::imwrite("res_debug/opacity_map.png", opacity_mat);
#endif
}

int getOptimalDCTSize(int n)
{
	return 2 * cv::getOptimalDFTSize((n + 1) / 2);
}

void FrequencyDec(const cv::Mat &fsrc, float threshold, float merge, cv::Mat &high, cv::Mat &low)
{
	int w = getOptimalDCTSize(fsrc.cols);
	int h = getOptimalDCTSize(fsrc.rows);
	float diag = std::sqrt(w*w + h * h);
	float beg = (threshold - merge) * diag;
	float end = (threshold + merge) * diag;

	cv::Mat padded;
	cv::copyMakeBorder(fsrc, padded, 0, h - fsrc.rows, 0, w - fsrc.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	std::vector<cv::Mat> planes;
	cv::split(padded, planes);
	std::vector<cv::Mat> hvec(planes.size()), lvec(planes.size());
	cv::Mat cos_transform;

	for (size_t i = 0; i < planes.size(); i++)
	{
		cv::dct(planes[i], cos_transform);

		cv::Mat cos_low(cos_transform.size(), cos_transform.type());
		cv::Mat cos_high(cos_transform.size(), cos_transform.type());

		for (int row = 0; row < padded.rows; row++) {
			float *p_cos = cos_transform.ptr<float>(row);
			float *p_high = cos_high.ptr<float>(row);
			float *p_low = cos_low.ptr<float>(row);

			for (int col = 0; col < padded.cols; col++) {
				float r = std::sqrt(col*col + row*row);
				float alpha = 0;
				if (r > beg && r < end) {
					alpha = 0.5f * (1.f - std::cos(3.14f * (r - beg) / (end - beg)));
				}
				else if (r >= end) {
					alpha = 1.f;
				}

				p_low[col] = (1.f - alpha) * p_cos[col];
				p_high[col] = alpha * p_cos[col];
			}
		}

		cv::dct(cos_high, hvec[i], cv::DftFlags::DCT_INVERSE);
		cv::dct(cos_low, lvec[i], cv::DftFlags::DCT_INVERSE);

		hvec[i] = hvec[i](cv::Rect(0, 0, fsrc.cols, fsrc.rows));
		lvec[i] = lvec[i](cv::Rect(0, 0, fsrc.cols, fsrc.rows));
	}

	cv::merge(hvec, high);
	cv::merge(lvec, low);

#if 0
	cv::imwrite("res_debug/fr_high.png", 255.0 * high);
	cv::imwrite("res_debug/fr_low.png", 255.0 * low);
	cv::imwrite("res_debug/fr_low_high.png", 255.0 * (low + high));
#endif
}

class FlowModel
{
public:
	FlowModel() {}
	FlowModel(cv::Mat &img, cv::Mat &velocity_map, cv::Mat &opacity_map, float frame_time)
		: m_source(img), m_velocity_map(velocity_map), m_Tframe(frame_time), m_opacity_map(opacity_map)
	{
		m_offset_map.create(img.size(), CV_32FC2);
		m_offset_map.setTo(cv::Scalar::all(0));
	}

	void Set(int Tstamp)
	{
		m_offset_map.setTo(cv::Scalar::all(0));
		int direction = Tstamp < 0 ? 1 : -1;

		for (int i = 0; i <= std::abs(Tstamp); i++) {

			cv::Mat offset_prev = m_offset_map.clone();

			for (int i = 0; i < m_offset_map.rows; i++) {
				cv::Vec2f *p_vec = m_velocity_map.ptr<cv::Vec2f>(i);
				cv::Vec2f *p_offset = m_offset_map.ptr<cv::Vec2f>(i);

				for (int j = 0; j < m_offset_map.cols; j++) {
					cv::Vec2f delta = direction * m_Tframe * p_vec[j];

					cv::Vec2f sample = offset_prev.at<cv::Vec2f>(i + delta[1], j + delta[0]);
					p_offset[j] = sample + delta;
				}
			}
		}
	}

	void GetNext(cv::Mat &dst, int frame = 0)
	{
		for (int i = 0; i < m_velocity_map.rows; i++) {
			cv::Vec2f *p_vec = m_velocity_map.ptr<cv::Vec2f>(i);
			cv::Vec3f *p_dst = dst.ptr<cv::Vec3f>(i);
			cv::Vec3f *p_src = m_source.ptr<cv::Vec3f>(i);

			for (int j = 0; j < m_offset_map.cols; j++) {
				cv::Vec2f delta = -m_Tframe * p_vec[j];

				float x = j + frame * delta[0];
				float y = i + frame * delta[1];

				if (x < 0 || x >= dst.cols || y < 0 || y >= dst.rows) {
					p_dst[j] = p_src[j];
				}
				//else if (p_vec[j] != cv::Vec2f() && m_velocity_map.at<cv::Vec2f>(y, x) == cv::Vec2f()) {
				//	p_dst[j] = p_src[j];
				//}
				else {
					cv::Vec3f color1, color2;
					//BilinInterp(m_source, x, y, &color1[0]);
					color1 = m_source.at<cv::Vec3f>(y, x);
					color2 = p_src[j];

					if (color1 != cv::Vec3f()) {
						float k = m_opacity_map.at<float>(y, x);
						p_dst[j] = (1 - k) * color2 + k * color1;
					}
				}
			}
		}
	}


public:
	cv::Mat m_source;
	cv::Mat m_offset_map;
	cv::Mat m_velocity_map;
	cv::Mat m_opacity_map;
	float m_Tframe;
};

inline cv::Vec3f color_blend(const cv::Vec3f &p1, const cv::Vec3f &p2, const float &x, const float &max_val)
{
	cv::Vec3f dst;

	/// сильный пиксель по трём каналам
	for (int i = 0; i < 3; i++) {
		float v = std::abs(p2[i]) / max_val;
		float p = (v >= 0.5f) ? (-1.6f * v + 1.8f) : (-8.f * v + 5.f);
		float k = std::pow(x, p);
		dst[i] = (1 - k) * p1[i] + k * p2[i];
	}

	return dst;
}


GLSLProgram glProgram;
Texture2D high_tex;
Texture2D offset_tex1;
Texture2D low_tex;
Texture2D velocity_tex1;
Texture2D opacity_tex1;

float g_Tframe;
float g_frame1, g_frame2;
float g_k, g_max_del;
cv::Mat g_frame;


int width, height;
int var = 0;

std::string vert_shader = R"glsl(
	#version 430

	layout (location=0) in vec2 VertexPosition;
	layout (location=1) in vec2 VertexText;

	uniform mat4 ModelViewMatrix;
	uniform mat3 NormalMatrix;
	uniform mat4 MVP;

	out vec2 TextCoord;

	void main()
	{
		TextCoord = VertexText;
		gl_Position = vec4(VertexPosition, 0.0, 1.0);
	}
)glsl";

//void GetNext(cv::Mat &dst, int frame = 0)
//{
//	cv::Mat offset_prev = m_offset_map.clone();
//	for (int i = 0; i < m_offset_map.rows; i++) {
//		cv::Vec2f *p_vec = m_velicity_map.ptr<cv::Vec2f>(i);
//		cv::Vec2f *p_offset = m_offset_map.ptr<cv::Vec2f>(i);
//		cv::Vec3f *p_dst = dst.ptr<cv::Vec3f>(i);
//		cv::Vec3f *p_src = m_source.ptr<cv::Vec3f>(i);
//		for (int j = 0; j < m_offset_map.cols; j++) {
//			cv::Vec2f delta = -m_Tframe * p_vec[j];
//			float x = j + frame * delta[0];
//			float y = i + frame * delta[1];
//
//			if (x < 0 || x >= dst.cols || y < 0 || y >= dst.rows) {
//				p_dst[j] = p_src[j];
//			}
//			else if (p_vec[j] != cv::Vec2f() && m_velicity_map.at<cv::Vec2f>(y, x) == cv::Vec2f()) {
//				p_dst[j] = p_src[j];
//			}
//			else {
//				cv::Vec3f color1, color2;
//				BilinInterp(m_source, x, y, &color1[0]);
//				color2 = p_src[j];
//
//				if (color1 != cv::Vec3f()) {
//					float k = m_opacity_map.at<float>(y, x);
//					p_dst[j] = (1 - k) * color2 + k * color1;
//				}
//			}
//		}
//	}
//}
//inline cv::Vec3f color_blend(const cv::Vec3f &p1, const cv::Vec3f &p2, const float &x, const float &max_val)
//{
//	cv::Vec3f dst;
//
//	/// сильный пиксель по трём каналам
//	for (int i = 0; i < 3; i++) {
//		float v = std::abs(p2[i]) / max_val;
//		float p = (v >= 0.5f) ? (-1.6f * v + 1.8f) : (-8.f * v + 5.f);
//		float k = std::pow(x, p);
//		dst[i] = (1 - k) * p1[i] + k * p2[i];
//	}
//
//	return dst;
//}
//for (int row = 0; row < dst.rows; row++) {
//	cv::Vec3f *p_w0 = wave0.ptr<cv::Vec3f>(row);
//	cv::Vec3f *p_w1 = wave1.ptr<cv::Vec3f>(row);
//	cv::Vec3f *p_dst = dst.ptr<cv::Vec3f>(row);
//
//	for (int col = 0; col < dst.cols; col++) {
//		cv::Vec3f f0 = p_w0[col];
//		cv::Vec3f f1 = p_w1[col];
//
//		float k = 1.f / (Nloop - 1.f) * i;
//		p_dst[col] += color_blend(f0, f1, k, maxVal);
//	}
//}

std::string frag_shader = R"glsl(
	#version 430

	in vec2 TextCoord;

	uniform sampler2D high;
	uniform sampler2D low;
	uniform sampler2D velocity_map;
	uniform sampler2D opacity_map;
	uniform sampler2D source;

	uniform vec2 viewPort;

	uniform float tframe;
	uniform float frame1;
	uniform float frame2;
	uniform float k;
	uniform float max_del;

	layout (location = 0) out vec4 FragColor;

	vec3 color_blend(vec4 p1, vec4 p2, float x, float max_del) {
		vec3 ret;
		vec4 v = abs(p2) * max_del;		
		for (int i = 0; i < 3; i++)
		{
			float p = (v[i] >= 0.5) ? (-1.6 * v[i] + 1.8) : (-8.0 * v[i] + 5.0);
			float k = pow(x, p);
			ret[i] = (1.0 - k) * p1[i] + k * p2[i];
		}
		return ret;
	}

	vec4 flow(float frame) {
		vec2 cur = gl_FragCoord.xy;
		ivec2 icur = ivec2(cur);
		vec2 delta = tframe * vec2(texelFetch(velocity_map, icur, 0));
		vec2 xy = cur + frame * delta;
		ivec2 ixy = ivec2(xy);
		xy.x *= viewPort.x;
		xy.y *= viewPort.y;
		if (xy.x < 0.0 || xy.x > 1.0 || xy.y < 0.0 || xy.y > 1.0) {
			return texelFetch(high, icur, 0);
		}
		vec4 color1 = texelFetch(high, ixy, 0);
		vec4 color2 = texelFetch(high, icur, 0);
		float k = texelFetch(opacity_map, ixy, 0)[0];
		return color2 * (1.0 - k) + color1 * k;
		//return color1;
	}

	void main() { 
		vec4 wave0 = flow(frame1);
		vec4 wave1 = flow(frame2);
		vec3 lowvec = vec3(texelFetch(low, ivec2(gl_FragCoord.xy), 0));
		FragColor = vec4(lowvec + color_blend(wave0, wave1, k, max_del), 1.0);
		//FragColor = wave0;
		//FragColor = texture(low, TextCoord) + wave1;
	}
)glsl";



void display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	std::vector<float> positionData = {
		-1, -1,
		 1, -1,
		 1,  1,
		-1,  1
	};
	std::vector<float> textureData = {
		0, 0,
		1, 0,
		1, 1,
		0, 1,
	};

	glProgram.setAtribute(0, positionData, 2);
	glProgram.setAtribute(1, textureData, 2);
	glProgram.setUniform("velocity_map", 3);
	glProgram.setUniform("opacity_map", 4);
	glProgram.setUniform("high", 1);
	glProgram.setUniform("low", 2);
	glProgram.setUniform("tframe", g_Tframe);
	glProgram.setUniform("frame1", g_frame1);
	glProgram.setUniform("frame2", g_frame2);
	glProgram.setUniform("k", g_k);
	glProgram.setUniform("max_del", g_max_del);
	glProgram.setUniform("viewPort", glm::vec2(1.f / (width - 1), 1.f / (height - 1)));
	glProgram.draw(GL_QUADS, 0, positionData.size());

	cv::Mat frame(height, width, CV_32FC4);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, frame.data);
	glutSwapBuffers();
	cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
	cv::flip(frame, frame, 0);
	//cv::imwrite("frames/" + std::to_string(pos++ % 60) + "_frame.png", frame * 200);
	frame.copyTo(g_frame);
}


void init(void) {
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
	glClearColor(1, 0, 0, 1);

	glProgram.PrintGPUVersion();
	glProgram.compileShader(vert_shader, GLSLShader::VERTEX);
	glProgram.compileShader(frag_shader, GLSLShader::FRAGMENT);
	glProgram.link();
	glProgram.use();
	glProgram.validate();
	glProgram.findUniformLocations();
}


void setTexture(Texture2D& tex, cv::Mat& img, GLenum slot = GL_TEXTURE1, bool switch_channels = true) {
	cv::Mat tx;
	if (img.channels() == 3) {
		img.copyTo(tx);
	}
	else if (img.channels() == 1) {
		cv::merge(std::vector<cv::Mat>({ img, img, img }), tx);
	}
	else {
		tx = _rendertestmask(img);
	}
	ASSERT(tx.channels() == 3);
	if (switch_channels) {
		cv::cvtColor(tx, tx, cv::COLOR_BGR2RGBA);
	}
	else {
		cv::cvtColor(tx, tx, cv::COLOR_RGB2RGBA);
	}
	ASSERT(tx.type() == 29 || tx.type() == 24);
	ASSERT(img.cols == width);
	ASSERT(img.rows == height);
	cv::flip(tx, tx, 0);
	if (tx.type() == 29) tex.createColorTexture(tx, GL_LINEAR, GL_LINEAR);
	if (tx.type() == 24) tex.createColorTexture(tx, GL_LINEAR, GL_LINEAR, GL_UNSIGNED_BYTE);
	tex.bind(slot);
}


void PhotoLoop(cv::Mat &src, cv::Mat &mask, cv::Mat &opacity_map, cv::Mat &high, cv::Mat &low, cv::Mat &field_map, std::string out_name, float Tloop,
	int argc, char** argv)
{
	const float fps = 24.f;
	const float Tframe = 1.f / fps;
	const float Nloop = std::floor(Tloop / Tframe);

	cv::imwrite("HIGH.png", high * 300);

	FlowModel flow0(high, field_map, opacity_map, Tframe);
	FlowModel flow1(high, field_map, opacity_map, Tframe);
	cv::Mat wave0(src.size(), src.type()), wave1(src.size(), src.type());
	cv::Mat dst(src.size(), src.type());
	cv::Mat frame(src.size(), CV_8UC3);
	std::vector<cv::Mat> video;

	double Mmin, Mmax;
	cv::minMaxLoc(high, &Mmin, &Mmax);
	float maxVal = std::max(std::abs(Mmin), Mmax);
	cv::Mat img;
	src.copyTo(img);

#if 1
	g_Tframe = Tframe * 10;
	g_max_del = 1.f / maxVal;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
	glutInitWindowSize(src.cols, src.rows);
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	
	
	width = src.cols;
	height = src.rows;
	
	glutCreateWindow("OpenGL Render");

	if (GLEW_OK != glewInit()) {
		std::cout << "Couldn't initialize GLEW" << std::endl;
		exit(0);
	}

	glutDisplayFunc(display);
	glutIdleFunc(display);

	///glutReshapeFunc(reshape);
	
	//setTexture(gl_src_image, img, GL_TEXTURE1);
	setTexture(high_tex, high, GL_TEXTURE1);
	setTexture(low_tex, low, GL_TEXTURE2);
	cv::Mat vel = flow0.m_velocity_map * 255;
	setTexture(velocity_tex1, vel, GL_TEXTURE3);
	setTexture(opacity_tex1, flow0.m_opacity_map, GL_TEXTURE4);
	init();
	//glutMainLoop();
	
#endif
	for (int i = 0; i < Nloop; i++) {
		std::cout << i + 1 << " of " << Nloop << std::endl;
#ifdef __CPU
		flow0.GetNext(wave0, i);
		flow1.GetNext(wave1, -Nloop + i);
		low.copyTo(dst);
		for (int row = 0; row < dst.rows; row++) {
			cv::Vec3f *p_w0 = wave0.ptr<cv::Vec3f>(row);
			cv::Vec3f *p_w1 = wave1.ptr<cv::Vec3f>(row);
			cv::Vec3f *p_dst = dst.ptr<cv::Vec3f>(row);

			for (int col = 0; col < dst.cols; col++) {
				cv::Vec3f f0 = p_w0[col];
				cv::Vec3f f1 = p_w1[col];

				float k = 1.f / (Nloop - 1.f) * i;
				p_dst[col] += color_blend(f0, f1, k, maxVal);
			}
		}
#endif
#ifdef __GPU
		g_frame1 = i;
		g_frame2 = -Nloop + i;
		g_k = 1.f / (Nloop - 1.f) * i;
		glProgram.setUniform("velocity_map", 3);
		glProgram.setUniform("opacity_map", 4);
		glProgram.setUniform("high", 1);
		glProgram.setUniform("low", 2);
		glProgram.setUniform("tframe", g_Tframe);
		glProgram.setUniform("frame1", g_frame1);
		glProgram.setUniform("frame2", g_frame2);
		glProgram.setUniform("k", g_k);
		glProgram.setUniform("max_del", g_max_del);
		glProgram.setUniform("viewPort", glm::vec2(width - 1, height - 1));
		//glutMainLoopEvent();
		display();
		g_frame.copyTo(dst);
#endif
		dst.convertTo(frame, CV_8UC3, 255.0);
		video.push_back(frame.clone());
	}

	///init writer
	VideoWriterMemory writer;
	IVideo_Encoder *encoder = video_encoder_create();
	encoder->Init(&writer, src.cols, src.rows, fps, 4000000);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < video.size(); j++) {
			unsigned char *p = video[j].ptr(0, 0);
			encoder->Addframe(p, video[j].step, 1);
		}
	}

	encoder->Finalize();
	encoder->Destroy();

	uchar *d;
	size_t len;
	writer.GetBuffer(d, len);
	std::ofstream out(out_name, std::ios::binary);
	out.write((char*)d, len);
	out.close();
	delete d;
}

void reshape(int width, int height) {
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0);
	glMatrixMode(GL_MODELVIEW);
}

void trackbar()
{
	cv::namedWindow("trackbar");
	cv::resizeWindow("trackbar", 500, 500);
	cv::createTrackbar("height", "trackbar", &var, 100);
	while (true)
	{
		cv::waitKey();
	}
}


int main(int argc, char **argv)
{
	Material material = Material::HAIR;

	dlib::frontal_face_detector	dlib_detector;
	dlib::shape_predictor pose_model;


	if (material == Material::HAIR) {
		dlib_detector = dlib::get_frontal_face_detector();
		try { 
			dlib::deserialize("res/shape_predictor_68_face_landmarks.dat") >> pose_model; 
		}
		catch (...) { 
			return -1; 
		}
	}

	std::filesystem::path dir("C:/Users/User/Desktop/flow_animation/hair");
	std::filesystem::directory_iterator it(dir), end;



	for (int count = 0; it != end; it++) {
		cv::Mat image = cv::imread(it->path().string());
		if (image.empty()) continue;

#ifdef __GPU
		std::string fname = it->path().filename().string() + "_gpu";
#else
		std::string fname = it->path().filename().string();
#endif
		std::string out_name = "results/" + fname + ".mp4";

		switch (material) {
		case Material::HAIR:
		{
			std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlib::cv_image<dlib::bgr_pixel>(image));
			if (faces_dlib.empty()) continue;
			dlib::full_object_detection	points = pose_model(dlib::cv_image<dlib::bgr_pixel>(image), faces_dlib[0]);


			cv::Vec2f direction(points.part(33).x() - points.part(27).x(), points.part(33).y() - points.part(27).y());
			direction = 0.042f * faces_dlib[0].height() * direction / cv::norm(direction);

			cv::Mat fimg;
			image.convertTo(fimg, CV_32FC3, 1 / 255.0);

			cv::Mat mask;
			int res = GetMask(image, mask, material);
			cv::imwrite("mask.png", mask);
			if (res != 0) return -1;

			cv::Mat velocity_field;
			cv::Mat opacity_map;
			std::vector<cv::Point2f> contours_points, normals_points;
			CreateVectorField(mask, opacity_map, velocity_field, direction, contours_points, normals_points, Material::HAIR);

			cv::Mat high, low;
			FrequencyDec(fimg, 0.07f, 0.04f, high, low);

			float Tloop = 2.5f;
			
			
			
			PhotoLoop(fimg, mask, opacity_map, high, low, velocity_field, out_name, Tloop, argc, argv);
			break;
		}
		case Material::WATER:
		{
			cv::Vec2f direction(image.cols * 0.015f, 0);
			
			cv::Mat fimg;
			image.convertTo(fimg, CV_32FC3, 1 / 255.0);

			cv::Mat mask;
			int res = GetMask(image, mask, material);
			if (res != 0) return -1;

			cv::Mat velocity_field;
			cv::Mat opacity_map;
			std::vector<cv::Point2f> contours_points, normals_points;
			CreateVectorField(mask, opacity_map, velocity_field, direction, contours_points, normals_points, Material::HAIR);

			cv::Mat high, low;
			FrequencyDec(fimg, 0.04f, 0.04f, high, low);

			float Tloop = 2.5f;
			PhotoLoop(fimg, mask, opacity_map, high, low, velocity_field, out_name, Tloop, argc, argv);

			break;
		}
		}


		std::cout << ++count << "images processed" << std::endl;
	}

	return 0;
}
