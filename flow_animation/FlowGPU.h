#pragma once
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include "render/GLSLProgram.h"
#include "render/Texture.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include "NvEncoder/NvPipe.h"
#include <fstream>
#if defined( _WIN32 )
#define ASSERT(x) if(!(x)) __debugbreak()
#else
#define ASSERT(x)
#endif

struct cdt_struct {
	float maxdel;
	cv::Mat high;
	cv::Mat low;
};

struct Timer {
	std::chrono::time_point<std::chrono::steady_clock> startp, endp;
	std::chrono::duration<float> duration;
	float ms[4];

	Timer() {
		startp = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 4; i++) ms[i] = 0.0;
	}
	~Timer() {
		//end();
	}

	void start() {
		startp = std::chrono::high_resolution_clock::now();
	}

	void end() {
		endp = std::chrono::high_resolution_clock::now();
		duration = endp - startp;
		float ms = duration.count() * 1000.0f;
		std::cout << "duration: " << ms << " ms\n";
	}

	float elapsed() {
		duration = std::chrono::high_resolution_clock::now() - startp;
		float ms = duration.count() * 1000.0f;
		start();
		return ms;
	}

	float processTime() {
		return duration.count() * 1000.0f;
	}
};

class FlowGPU
{
public:
	FlowGPU() {}
	~FlowGPU() { NvPipe_Destroy(NVencoder); m_ofs.close(); }
	FlowGPU(int argc, char** argv, cv::Mat& velocity, cv::Mat& opacity, cv::Mat& high, cv::Mat& low, float Tframe, int num_fr);
	void setTexture(Texture2D& tex, cv::Mat& img, GLenum slot = GL_TEXTURE1, bool add_channels = true, bool switch_channels = true);
	static cv::Mat _rendertestmask(cv::Mat& testimage);
	void display();
	void init(int argc, char** argv, cv::Mat& velocity, cv::Mat& opacity, cv::Mat& high, cv::Mat& low);
	void updateSpeed(int Nloop, float Trame);
private:
	GLSLProgram glProgram;
	Texture2D high_tex;
	Texture2D offset_tex1;
	Texture2D low_tex;
	Texture2D velocity_tex1;
	Texture2D opacity_tex1;
	GLuint serverFBO;
	GLuint serverColorTex;
	NvPipe* NVencoder;
	std::ofstream m_ofs;

	std::vector<Texture2D>f_texs;

	float m_Tframe, m_maxVal_frac;
	int m_width, m_height;
	float num_frames;
	float m_frame1, m_frame2, m_k, m_i;
	std::mutex m_mutex;

	inline static const std::string vert_shader = R"glsl(
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

	inline static const std::string frag_shader = R"glsl(
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
	uniform float thr;
	uniform float tb_gammaBound;

	layout (location = 0) out vec4 FragColor;

	vec3 color_blend(vec4 p1, vec4 p2, float x, float max_del) {
		vec3 ret;
		vec4 v = abs(p2) * max_del;		
		for (int i = 0; i < 3; i++)
		{	
			float add;
			float mul;
			if (v[i] < thr) {
				add = tb_gammaBound;
				mul = (1.0 - tb_gammaBound) / thr;
			}
			else {
				add = (thr / tb_gammaBound - 1.0) / (thr - 1.0);
				mul = 1.0 / tb_gammaBound - add;
			}
			float P = mul * v[i] + add;

			//float p = (v[i] >= 0.5) ? (-1.6 * v[i] + 1.8) : (-8.0 * v[i] + 5.0);
			float k = pow(x, P);
			ret[i] = (1.0 - k) * p1[i] + k * p2[i];
		}
		return ret;
	}


	vec4 flow(float frame) {
		vec2 cur = gl_FragCoord.xy;
		ivec2 icur = ivec2(cur);
		vec4 delta = tframe * texture(velocity_map, TextCoord);
		vec2 xy = cur;
		xy.x -= frame * delta[0];
		xy.y -= frame * delta[1];
		ivec2 ixy = ivec2(xy);
		xy.x *= viewPort.x;
		xy.y *= viewPort.y;
		if (xy.x < 0.0 || xy.x > 1.0 || xy.y < 0.0 || xy.y > 1.0) {
			return texture(high, TextCoord);
		}
		if (vec2(delta) != vec2(0.0) && vec2(texture(velocity_map, xy)) == vec2(0.0)) {
			return texture(high, TextCoord);
		}	
		vec4 color1 = texture(high, xy);
		vec4 color2 = texture(high, TextCoord);
		float k = texture(opacity_map, xy)[0];
		return color2 * (1.0 - k) + color1 * k;
	}

	void main() { 
		vec4 wave0 = flow(frame1);
		vec4 wave1 = flow(frame2);
		vec3 lowvec = vec3(texture(low, TextCoord));
		FragColor = (vec4(lowvec + color_blend(wave0, wave1, k, max_del), 1.0));
		//FragColor = texture(low, TextCoord);
	}
)glsl";

};

