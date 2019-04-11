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
#include "render/GLSLProgram.h"
#include "render/Texture.hpp"
#include <thread>
#include <filesystem>

///dlib
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

template<class Type>
void BilinInterp(const cv::Mat &I, double x, double y, Type *dst)
{
	int x1 = (int)std::floor(x);
	int y1 = (int)std::floor(y);
	int x2 = (int)std::ceil(x);
	int y2 = (int)std::ceil(y);
	if (x1 < 0 || x2 >= I.cols || y1 < 0 || y2 >= I.rows) return;

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

int GetMask(const cv::Mat &src, cv::Mat &dst)
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

	curl_formadd(&formpost,
		&lastptr,
		CURLFORM_COPYNAME, "detector",
		CURLFORM_COPYCONTENTS, "hair",
		CURLFORM_END
	);

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
		//curl_easy_setopt(curl, CURLOPT_URL, "http://67.228.246.51:8056/forward");
		curl_easy_setopt(curl, CURLOPT_URL, "http://admin:show-me-viewer@exp.ws.pho.to/viewer/detect-blob");
		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result_chunk);

		res = curl_easy_perform(curl);

		if (res == CURLE_OK) {
			std::vector<uchar> buf(result_chunk.memory, result_chunk.memory + result_chunk.size);
			dst = cv::imdecode(buf, cv::ImreadModes::IMREAD_UNCHANGED);

			if (dst.empty()) {
				std::cout << std::string(buf.begin(), buf.end()) << std::endl;
				return -1;
			}

			//for (int i = 0; i < dst.rows; i++) {
			//	for (int j = 0; j < dst.cols; j++) {
			//		if (dst.at<uchar>(i, j) == 5) dst.at<uchar>(i, j) = 255;
			//		else dst.at<uchar>(i, j) = 0;
			//	}
			//}
		}

		curl_easy_cleanup(curl);
		curl_formfree(formpost);

		free(result_chunk.memory);
		curl_global_cleanup();
	}

	return res;
}

enum class Material
{
	WATER, HAIR, SKY, TREE
};

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



void CreateVectorField(const cv::Mat &mask, cv::Mat &dst, cv::Point2f dir, std::vector<cv::Point2f> &contour_points, std::vector<cv::Point2f> &normals_points, Material material)
{
	cv::Mat distance(mask.size(), CV_32FC1);
	cv::distanceTransform(mask, distance, cv::DistanceTypes::DIST_L2, cv::DIST_MASK_PRECISE);

	dst.create(mask.size(), CV_32FC2);
	dst.setTo(cv::Scalar::all(0));

	float offset_koeff = 0.04;// 0.019f;
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

			for (int j = 0; j < dst.cols; j++)
			{
#if 0
				p_dst[j] = dir * std::min(p_dist[j] / offset, 1.f);
#else
				if (p_dist[j] >= offset) {
					p_dst[j] = dir;
				}
				else {
					float d = dir_norm.x * p_norm[j][0] + dir_norm.y * p_norm[j][1];
					float p = 0.45f * d + 0.55f;
					float x = p_dist[j] / offset;
					float alpha = std::pow(x, p);
					p_dst[j] = alpha * dir;
				}
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

#if 0
	cv::Mat vec[2];
	cv::split(dst, vec);
	cv::magnitude(vec[0], vec[1], vec[0]);
	cv::Mat_<uchar> show(200 / cv::norm(dir) * vec[0]);
	cv::imwrite("VectorField.png", show);
	cv::imshow("test", show);
	cv::waitKey();
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
	cv::imshow("high", high);
	cv::imshow("low", low);
	cv::imshow("low+high", low + high);
	cv::waitKey();
#endif
}

class FlowModel
{
public:
	FlowModel(cv::Mat &img, cv::Mat &velocity_map, float frame_time)
		: m_source(img), m_velicity_map(velocity_map), m_Tframe(frame_time)
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
				cv::Vec2f *p_vec = m_velicity_map.ptr<cv::Vec2f>(i);
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
		cv::Mat offset_prev = m_offset_map.clone();

		for (int i = 0; i < m_offset_map.rows; i++) {
			cv::Vec2f *p_vec = m_velicity_map.ptr<cv::Vec2f>(i);
			cv::Vec2f *p_offset = m_offset_map.ptr<cv::Vec2f>(i);
			cv::Vec3f *p_dst = dst.ptr<cv::Vec3f>(i);

			for (int j = 0; j < m_offset_map.cols; j++) {
				cv::Vec2f delta = -m_Tframe * p_vec[j];
				
#if 0
				cv::Vec2f sample = offset_prev.at<cv::Vec2f>(i + delta[1], j + delta[0]);
				p_offset[j] = delta + sample;

				BilinInterp(m_source, j + p_offset[j][0], i + p_offset[j][1], &p_dst[j][0]);
#else
				float x = j + frame * delta[0];
				float y = i + frame * delta[1];
				if (p_vec[j] != cv::Vec2f() &&
					x >= 0 && x < dst.cols && y >= 0 && y < dst.rows &&
					m_velicity_map.at<cv::Vec2f>(y, x) == cv::Vec2f()) {
					p_dst[j] = cv::Vec3f();
				}
				else {
					BilinInterp(m_source, x, y, &p_dst[j][0]);
				}
#endif
			}
		}
	}


public:
	cv::Mat m_source;
	cv::Mat m_offset_map;
	cv::Mat m_velicity_map;
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

void BilinInterp2(const cv::Mat &I, double x, double y, cv::Vec3f &dst)
{
	int x1 = (int)std::floor(x);
	int y1 = (int)std::floor(y);
	int x2 = (int)std::ceil(x);
	int y2 = (int)std::ceil(y);
	if (x1 < 0 || x2 >= I.cols || y1 < 0 || y2 >= I.rows) return;

	const float *p1 = I.ptr<float>(y1, x1);
	const float *p2 = I.ptr<float>(y1, x2);
	const float *p3 = I.ptr<float>(y2, x1);
	const float *p4 = I.ptr<float>(y2, x2);

	for (int i = 0; i < 3; i++)
	{
		float c1 = p1[i] + (p2[i] - p1[i]) * (x - x1);
		float c2 = p3[i] + (p4[i] - p3[i]) * (x - x1);
		dst[i] = c1 + (c2 - c1) * (y - y1);
	}
}

void MirrorImage(cv::Mat &image, std::vector<cv::Point2f> &contour_points, std::vector<cv::Point2f> &normals_points, float dist)
{
	struct Tri {
		cv::Point2f v1, v2, v3;
	};

	std::vector<Tri> src, dst;

	for (size_t i = 0; i < contour_points.size(); i++) {
		const cv::Point2f &v1 = contour_points[i];
		const cv::Point2f &v2 = contour_points[(i + 1) % contour_points.size()];

		const cv::Point2f &n1 = normals_points[i];
		const cv::Point2f &n2 = normals_points[(i + 1) % contour_points.size()];

		cv::Point v1_top = v1 + dist * n1;
		cv::Point v2_top = v2 + dist * n2;
		cv::Point v1_down = v1 - dist * n1;
		cv::Point v2_down = v2 - dist * n2;

		Tri tri1_top{ v1, v2, v1_top };
		Tri tri2_top{ v1, v2_top, v1_top };
		Tri tri2_down{ v1, v2_down, v1_down };
		Tri tri1_down{ v1, v2, v1_down };

		dst.push_back(tri1_top);
		dst.push_back(tri2_top);
		src.push_back(tri2_down);
		src.push_back(tri1_down);
	}

	for (size_t t = 0; t < src.size(); t++)
	{
		const Tri &s = src[t];
		const Tri &d = dst[t];

		cv::Mat1d A(3, 3);  A << d.v1.x, d.v2.x, d.v3.x, d.v1.y, d.v2.y, d.v3.y, 1, 1, 1;
		cv::Mat1d B(3, 3);  B << s.v1.x, s.v2.x, s.v3.x, s.v1.y, s.v2.y, s.v3.y, 1, 1, 1;

		cv::Mat1d M = B * A.inv();
		double *affine = M.ptr<double>();

		int xmax = std::ceil(std::max(std::max(d.v1.x, d.v2.x), d.v3.x));
		int ymax = std::ceil(std::max(std::max(d.v1.y, d.v2.y), d.v3.y));
		int xmin = std::floor(std::min(std::min(d.v1.x, d.v2.x), d.v3.x));
		int ymin = std::floor(std::min(std::min(d.v1.y, d.v2.y), d.v3.y));
		if (xmax > image.cols - 1) xmax = image.cols - 1;
		if (ymax > image.rows - 1) ymax = image.rows - 1;
		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;

		for (int i = ymin; i <= ymax; i++)
		{
			cv::Vec3f *p_dst = image.ptr<cv::Vec3f>(i);

			for (int j = xmin; j <= xmax; j++)
			{
				double x = affine[0] * j + affine[1] * i + affine[2];
				double y = affine[3] * j + affine[4] * i + affine[5];
				BilinInterp2(image, x, y, p_dst[j]);
			}
		}
	}
}

void PhotoLoop(cv::Mat &src, cv::Mat &mask, cv::Mat &high, cv::Mat &low, cv::Mat &field_map, std::string out_name, float Tloop)
{
	const float fps = 24.f;
	const float Tframe = 1.f / fps;
	const float Nloop = std::floor(Tloop / Tframe);

	FlowModel flow0(high, field_map, Tframe);
	FlowModel flow1(high, field_map, Tframe);
	cv::Mat wave0(src.size(), src.type()), wave1(src.size(), src.type());
	cv::Mat dst(src.size(), src.type());
	cv::Mat frame(src.size(), CV_8UC3);
	std::vector<cv::Mat> video;

	double Mmin, Mmax;
	cv::minMaxLoc(high, &Mmin, &Mmax);
	float maxVal = std::max(std::abs(Mmin), Mmax);

	flow0.Set(0);
	flow1.Set(-Nloop);

	for (int i = 0; i < Nloop; i++) {
		std::cout << i + 1 << " of " << Nloop << std::endl;

#if 0
		flow0.GetNext(wave0);
		flow1.GetNext(wave1);
#else
		flow0.GetNext(wave0, i);
		flow1.GetNext(wave1, -Nloop + 1 + i);
#endif

		low.copyTo(dst);

		for (int row = 0; row < dst.rows; row++) {
			cv::Vec3f *p_w0 = wave0.ptr<cv::Vec3f>(row);
			cv::Vec3f *p_w1 = wave1.ptr<cv::Vec3f>(row);
			cv::Vec3f *p_dst = dst.ptr<cv::Vec3f>(row);

			for (int col = 0; col < dst.cols; col++) {
				cv::Vec3f f0 = /*cv::Vec3f();*/ p_w0[col];
				cv::Vec3f f1 = /*cv::Vec3f();*/ p_w1[col];

				float k = 1.f / (Nloop - 1.f) * i;
				p_dst[col] += color_blend(f0, f1, k, maxVal);
			}
		}

		dst.convertTo(frame, CV_8UC3, 255.0);
		video.push_back(frame.clone());
		//cv::imwrite("frames/" + std::to_string(i) + ".png", frame);
	}


	///init writer
	VideoWriterMemory writer;
	IVideo_Encoder *encoder = video_encoder_create();
	encoder->Init(&writer, src.cols, src.rows, fps, 4000000);

	for (int i = 0; i < 1; i++) {
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


GLSLProgram glProgram;
Texture2D gl_src_image;
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

std::string frag_shader = R"glsl(
	#version 430

	in vec2 TextCoord;

	uniform sampler2D Text;

	layout (location = 0) out vec4 FragColor;

	void main() {
		FragColor = texture(Text, TextCoord);
	}
)glsl";

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	std::cout << var << std::endl;


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
	glProgram.setUniform("Text", 1);
	glProgram.draw(GL_QUADS, 0, positionData.size());

	glutSwapBuffers();

#if 1
	static int pos = 0;
	cv::Mat frame(height, width, CV_8UC4);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, frame.data);

	cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
	cv::flip(frame, frame, 0);
	cv::imwrite("frames/" + std::to_string(pos++ % 10) + "_frame.png", frame);
#endif
}

void init(void) {
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
	glClearColor(1,0,0,1);

	glProgram.PrintGPUVersion();
	glProgram.compileShader(vert_shader, GLSLShader::VERTEX);
	glProgram.compileShader(frag_shader, GLSLShader::FRAGMENT);
	glProgram.link();
	glProgram.use();
	glProgram.validate();
	glProgram.findUniformLocations();
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

int process(const cv::Mat &image, cv::Vec2f dir, std::string out_name)
{
	width = image.cols;
	height = image.rows;
#if 0
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("OpenGL Render");

	if (GLEW_OK != glewInit()) {
		std::cout << "Couldn't initialize GLEW" << std::endl;
		exit(0);
	}

	glutDisplayFunc(display);
	glutIdleFunc(display);
	///glutReshapeFunc(reshape);

	cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);
	cv::flip(img, img, 0);
	gl_src_image.createColorTexture(img, GL_LINEAR, GL_LINEAR);
	gl_src_image.bind(GL_TEXTURE1);
	init();

	std::thread(trackbar).detach();
	glutMainLoop();
#endif
	cv::Mat fimg;
	image.convertTo(fimg, CV_32FC3, 1 / 255.0);

#if 1
	cv::Mat mask;
	int res = GetMask(image, mask);
	if (res != 0) return -1;
	//cv::imwrite("m3.png", mask);
#else
	cv::Mat mask = cv::imread("m3.png", cv::ImreadModes::IMREAD_UNCHANGED);
#endif

	cv::Mat velocity_field;
	std::vector<cv::Point2f> contours_points, normals_points;
	CreateVectorField(mask, velocity_field, dir, contours_points, normals_points, Material::HAIR);

	cv::Mat high, low;
	FrequencyDec(fimg, 0.08f, 0.04f, high, low);

	float Tloop = 3.f;
	float dist = Tloop * cv::norm(dir);
	//MirrorImage(high, contours_points, normals_points, dist);

	PhotoLoop(fimg, mask, high, low, velocity_field, out_name, Tloop);
}

int main(int argc, char **argv)
{
	dlib::frontal_face_detector	dlib_detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	try {
		dlib::deserialize("res/shape_predictor_68_face_landmarks.dat") >> pose_model;
	}
	catch (...) {
		return -1;
	}

	std::filesystem::path dir("C:/Users/Ainur/Desktop/Data/TestImages");
	std::filesystem::directory_iterator it(dir), end;
	int count = 0;

	for (; it != end; it++) {
		cv::Mat image = cv::imread(it->path().string());
		//cv::Mat image = cv::imread("image2.jpg");
		if (image.empty()) continue;

		std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlib::cv_image<dlib::bgr_pixel>(image));
		if (faces_dlib.empty()) continue;
		dlib::full_object_detection	points = pose_model(dlib::cv_image<dlib::bgr_pixel>(image), faces_dlib[0]);
		

		cv::Vec2f direction(points.part(33).x() - points.part(27).x(), points.part(33).y() - points.part(27).y());
		direction = 0.039f * faces_dlib[0].height() * direction / cv::norm(direction);
		std::string fname = it->path().filename().string();
		std::string out_name = "results/" + fname + ".mp4";

		process(image, direction, out_name);
		//exit(0);
	std::cout << ++count << "images processed" << std::endl;
	}

	return 0;
}