#pragma once

#include <stdlib.h>
#include <string>
#include <cstring>


class IVideo_Writer {
public:
	virtual int write(unsigned char *buf, int buf_size) = 0;
    virtual long long seek(long long offset, int whence) = 0;
};

class IVideo_Encoder {
public:
    virtual int  Init(IVideo_Writer *, int width, int height, int fps, int bitrate /*=400000*/) = 0;
    virtual int  Addframe(unsigned char *rgb_data, int pitch, int duration/*=1*/) = 0;
    virtual int  Finalize() = 0;
    virtual void GetErrorDescription(char *buf, size_t len, int code) = 0;
    virtual void Destroy() = 0;
	virtual int	 AddAudio(std::string filename) = 0;
};

class VideoWriterMemory : public IVideo_Writer {
public:
	VideoWriterMemory()
	{
		m_buffer = NULL;
		m_buffer_size = 0;
		m_pos = 0;
	}

	virtual int write(unsigned char *buf, int size)
	{
		if (m_pos + size > m_buffer_size) {
			m_buffer_size = m_pos + size;
			m_buffer = (unsigned char*)realloc(m_buffer, m_buffer_size);
		}
		memcpy(m_buffer + m_pos, buf, size);
		m_pos += size;
		return size;
	}

	virtual long long seek(long long offset, int whence)
	{
		switch (whence) {
		case SEEK_SET:
			m_pos = (unsigned)offset;
			break;
		case SEEK_CUR:
			m_pos += (unsigned)offset;
			break;
		case SEEK_END:
			m_pos = m_buffer_size + (unsigned)offset;
			break;
		}
		if (m_pos > m_buffer_size)
		{
			m_buffer_size = m_pos;
			m_buffer = (unsigned char*)realloc(m_buffer, m_buffer_size);
		}
		return m_pos;
	}

	void GetBuffer(unsigned char *&data, size_t &size)
	{
		data = m_buffer;
		size = m_buffer_size;
	}

private:
	unsigned char *m_buffer;
	size_t         m_buffer_size;
	size_t         m_pos;
};

IVideo_Encoder *video_encoder_create();
