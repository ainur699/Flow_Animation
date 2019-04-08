#pragma once

#include <opencv2/core.hpp>



class FgVideoCapture {
public:
	FgVideoCapture() {
		m_position = 0;
		m_fps = 0;
	}

	int			ReadFromFile(std::string video_fname);
	cv::Mat		getFrame();
	int			size() const;
	void		setPos(uint pos);
	int			fps() { return m_fps; }
	bool		isOpened() const { return size(); }
	void		release() { frame_buf.clear(); }

private:
	int m_position;
	int m_fps;
	std::vector<cv::Mat> frame_buf;
};
