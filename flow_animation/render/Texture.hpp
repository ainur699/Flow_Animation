#pragma once
#include <opencv2/core.hpp>
#include <GL/glew.h>


//-----------------------------------------------------------------------------
class Texture
{
public:
	Texture();
	virtual ~Texture();

	virtual void updateImage() = 0;
	virtual void resize(int width, int height) = 0;
	virtual void bind(GLenum textureUnit) = 0;
	virtual void unbind(GLenum textureUnit) = 0;

	int		width() const { return m_width; }
	int		height() const { return m_height; }
	GLuint	id() const { return m_id; }
	GLenum	unit() const { return m_unit; }

protected:
	GLuint	m_id;
	GLenum	m_unit;
	GLint	m_width, m_height;
	GLint	m_internalFormat;
	GLint	m_magFilter;
	GLint	m_minFilter;
};


//-----------------------------------------------------------------------------
class Texture2D : public Texture
{
public:
	Texture2D() {}
	virtual ~Texture2D() {}

	void createColorTexture(const cv::Mat &image, GLint magFilter, GLint minFilter, GLenum type = GL_FLOAT, GLint internalFormat = GL_RGBA);
	void createColorTexture(int width, int height, GLint magFilter, GLint minFilter);
	void createDepthTexture(int width, int height, GLint magFilter, GLint minFilter);

	void updateImage();
	void resize(int width, int height);
	void bind(GLenum textureUnit);
	void unbind(GLenum textureUnit);

protected:
	void _create(GLvoid *data,
		GLint internalFormat,
		int width,
		int height,
		GLenum format,
		GLenum type,
		GLint magFilter,
		GLint minFilter);

private:
	cv::Mat m_image;
	GLenum	m_format;
	GLenum	m_type;
};


//-----------------------------------------------------------------------------
class Texture2DMultiSample : public Texture
{
public:
	Texture2DMultiSample() {}
	virtual ~Texture2DMultiSample() {}

	void createColorTexture(int width, int height, GLsizei samples, GLint magFilter, GLint minFilter);
	void createDepthTexture(int width, int height, GLsizei samples, GLint magFilter, GLint minFilter);

	void updateImage();
	void resize(int width, int height);
	void bind(GLenum textureUnit);
	void unbind(GLenum textureUnit);

protected:
	void _create(
		GLint internalFormat,
		int width,
		int height,
		GLsizei samples,
		GLint magFilter,
		GLint minFilter);

private:
	GLsizei m_samples;
};
