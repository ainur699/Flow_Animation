#include "FlowGPU.h"
#include <iostream>

FlowGPU::FlowGPU(int argc, char** argv, cv::Mat& velocity, cv::Mat& opacity, cv::Mat& high, cv::Mat& low, float Tframe): g_Tframe(Tframe) {
	width = high.cols;
	height = high.rows;

	double Mmin, Mmax;
	cv::minMaxLoc(high, &Mmin, &Mmax);
	float maxVal = std::max(std::abs(Mmin), Mmax);
	g_max_del = 1.f / maxVal;
	
	init(argc, argv, velocity, opacity, high, low);
}

cv::Mat FlowGPU::display(float frame1, float frame2, float k) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	static const std::vector<float> positionData = {
		-1, -1,
		 1, -1,
		 1,  1,
		-1,  1
	};
	static const std::vector<float> textureData = {
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
	glProgram.setUniform("frame1", frame1);
	glProgram.setUniform("frame2", frame2);
	glProgram.setUniform("k", k);
	glProgram.setUniform("max_del", g_max_del);
	glProgram.setUniform("viewPort", glm::vec2(1.f / (width - 1), 1.f / (height - 1)));
	glProgram.draw(GL_QUADS, 0, positionData.size());

	cv::Mat frame(height, width, CV_32FC4);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, frame.data);
	glutSwapBuffers();
	cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
	cv::flip(frame, frame, 0);
	return frame;
}


void FlowGPU::init(int argc, char** argv, cv::Mat& velocity, cv::Mat& opacity, cv::Mat& high, cv::Mat& low) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_MULTISAMPLE);
	glutInitWindowSize(width, height);
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glutCreateWindow("OpenGL Render");
	if (GLEW_OK != glewInit()) {
		std::cout << "Couldn't initialize GLEW" << std::endl;
		exit(0);
	}
	setTexture(high_tex, high, GL_TEXTURE1);
	setTexture(low_tex, low, GL_TEXTURE2);
	setTexture(velocity_tex1, velocity, GL_TEXTURE3);
	setTexture(opacity_tex1, opacity, GL_TEXTURE4);

	glEnable(GL_TEXTURE_2D);
	//glEnable(GL_DEPTH_TEST);
	glClearColor(0.f, 0.f, 0.f, 1.f);

	glProgram.PrintGPUVersion();
	glProgram.compileShader(vert_shader, GLSLShader::VERTEX);
	glProgram.compileShader(frag_shader, GLSLShader::FRAGMENT);
	glProgram.link();
	glProgram.use();
	glProgram.validate();
	glProgram.findUniformLocations();
}

#define __DEBUG
#if defined( _WIN32 )
#ifdef __DEBUG
#define ASSERT(x) if(!(x)) __debugbreak()
#else
#define ASSERT(x)
#endif
#endif
void FlowGPU::setTexture(Texture2D& tex, cv::Mat& img, GLenum slot, bool switch_channels) {
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


cv::Mat FlowGPU::_rendertestmask(cv::Mat& testimage)
{
	cv::Mat rendertestimage;
	std::vector<cv::Mat> channels;
	cv::split(testimage, channels);
	channels.push_back(cv::Mat::zeros(channels[0].rows, channels[0].cols, channels[0].type()));
	cv::merge(channels, rendertestimage);
	return rendertestimage;
}
