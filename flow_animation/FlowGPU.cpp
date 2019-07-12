#include "FlowGPU.h"
#include <iostream>

FlowGPU::FlowGPU(int argc, char** argv, cv::Mat& velocity, cv::Mat& opacity, cv::Mat& high, cv::Mat& low, float Tframe, int num_fr)
: m_Tframe(Tframe), num_frames(num_fr), m_frame1(0), m_frame2(-num_fr), m_k(0), m_i(0){
	m_width = high.cols;
	m_height = high.rows;

	double Mmin, Mmax;
	cv::minMaxLoc(high, &Mmin, &Mmax);
	float maxVal = std::max(std::abs(Mmin), Mmax);
	m_maxVal_frac = 1.f / maxVal;
	
	init(argc, argv, velocity, opacity, high, low);
	glProgram.setUniform("high", 1);
	glProgram.setUniform("low", 2); 
	glProgram.setUniform("velocity_map", 3);
	glProgram.setUniform("opacity_map", 4);
	glProgram.setUniform("tframe", m_Tframe);
	glProgram.setUniform("max_del", m_maxVal_frac);
	glProgram.setUniform("viewPort", glm::vec2(1.f / (m_width - 1), 1.f / (m_height - 1)));
}


void FlowGPU::updateSpeed(int Nloop, float Tframe) {
	m_Tframe = Tframe;
	if ((int)num_frames != Nloop) {
		m_i /= num_frames;
		m_i *= Nloop;
		num_frames = Nloop;
	}
}

cv::Mat FlowGPU::display() {
	Timer timer;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	static const std::vector<float> positionData = {
		-1, -1, 1, -1,
		1,  1, -1,  1
	};
	static const std::vector<float> textureData = {	
		0, 0, 1, 0,
		1, 1, 0, 1,
	};

	glProgram.setAtribute(0, positionData, 2);
	glProgram.setAtribute(1, textureData, 2);
	//i, -Nloop + i, 1.f / (Nloop - 1.f) * i
	extern int trackbar_Var;
	extern int trackbar_Tframe;
	updateSpeed(trackbar_Var + 2, 1.f / trackbar_Tframe);
	extern std::vector<std::vector<cdt_struct> > cdts;
	static int thresh = 0;
	static int merge = 0;
	extern int trackbar_cdt_threshold;
	extern int trackbar_cdt_merge;
	if (trackbar_cdt_threshold != thresh || trackbar_cdt_merge != merge) {
		cdt_struct& cur_cdt = cdts[trackbar_cdt_threshold][trackbar_cdt_merge];
		setTexture(high_tex, cur_cdt.high, GL_TEXTURE1, false);
		setTexture(low_tex, cur_cdt.low, GL_TEXTURE2, false);
		glProgram.setUniform("high", 1);
		glProgram.setUniform("low", 2);
		glProgram.setUniform("max_del", cur_cdt.maxdel);
	}
	glProgram.setUniform("frame1", m_i);
	glProgram.setUniform("frame2", -num_frames + m_i);
	glProgram.setUniform("tframe", m_Tframe);
	glProgram.setUniform("k", 1.f / (num_frames - 1.f) * m_i);
	extern int tb_gammaBound;
	extern int tb_powerThresh;
	glProgram.setUniform("thr", tb_powerThresh / 10.f);
	glProgram.setUniform("tb_gammaBound", (float) tb_gammaBound);

	glProgram.draw(GL_QUADS, 0, positionData.size());
	m_i = (int)(m_i + 1) % (int)num_frames;

	cv::Mat frame(m_height, m_width, CV_32FC4);
	glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, frame.data);
	glutSwapBuffers();
	cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
	cv::flip(frame, frame, 0);
	return frame;
}


void FlowGPU::init(int argc, char** argv, cv::Mat& velocity, cv::Mat& opacity, cv::Mat& high, cv::Mat& low) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_MULTISAMPLE);
	glutInitWindowSize(m_width, m_height);
	glViewport(0, 0, (GLsizei)m_width, (GLsizei)m_height);
	glutCreateWindow("OpenGL Render");
	if (GLEW_OK != glewInit()) {
		std::cout << "Couldn't initialize GLEW" << std::endl;
		exit(0);
	}
	setTexture(high_tex, high, GL_TEXTURE1, false);
	setTexture(low_tex, low, GL_TEXTURE2, false);
	setTexture(velocity_tex1, velocity, GL_TEXTURE3, true, false);
	setTexture(opacity_tex1, opacity, GL_TEXTURE4);

	glEnable(GL_TEXTURE_2D);
	//glEnable(GL_DEPTH_TEST);
	glClearColor(1.f, 0.f, 0.f, 1.f);

	glProgram.PrintGPUVersion();
	glProgram.compileShader(vert_shader, GLSLShader::VERTEX);
	glProgram.compileShader(frag_shader, GLSLShader::FRAGMENT);
	glProgram.link();
	glProgram.use();
	glProgram.validate();
	glProgram.findUniformLocations();
}


void FlowGPU::setTexture(Texture2D& tex, cv::Mat& img, GLenum slot, bool add_channels, bool switch_channels) {
	tex.remove();
	if (!add_channels || img.type() == 29) {
		tex.createColorTexture(img, GL_LINEAR, GL_LINEAR);
		tex.bind(slot);
		return;
	}

	cv::Mat tx;
	int ch = img.channels();
	switch (ch) {
	case 3:
		img.copyTo(tx);
		break;
	case 2:
		tx = _rendertestmask(img);
		break;
	case 1:
		cv::merge(std::vector<cv::Mat>({ img, img, img }), tx);
		break;
	default:
		ASSERT(0);
		break;
	}
	ASSERT(tx.channels() == 3);
	if (switch_channels) {
		cv::cvtColor(tx, tx, cv::COLOR_BGR2RGBA);
	}
	else {
		cv::cvtColor(tx, tx, cv::COLOR_RGB2RGBA);
	}
	ASSERT(tx.type() == 29);
	ASSERT(tx.cols == m_width);
	ASSERT(tx.rows == m_height);
	cv::flip(tx, tx, 0);
	tex.createColorTexture(tx, GL_LINEAR, GL_LINEAR);
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
