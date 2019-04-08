#include "Texture.hpp"
#include <limits>


//-----------------------------------------------------------------------------
Texture::Texture()
{
	m_id = std::numeric_limits<GLuint>::max();
	m_unit = std::numeric_limits<GLenum>::max();
}


//-----------------------------------------------------------------------------
Texture::~Texture()
{
	if (m_id != std::numeric_limits<GLuint>::max()) glDeleteTextures(1, &m_id);
}


//-----------------------------------------------------------------------------
void Texture2D::createColorTexture(const cv::Mat &image, GLint magFilter, GLint minFilter)
{
	m_image = image.clone();
	_create(m_image.data, GL_RGBA, m_image.cols, m_image.rows, GL_RGBA, GL_UNSIGNED_BYTE, magFilter, minFilter);
}


//-----------------------------------------------------------------------------
void Texture2D::createColorTexture(int width, int height, GLint magFilter, GLint minFilter)
{
	_create(0, GL_RGBA, width, height, GL_RGBA, GL_UNSIGNED_BYTE, magFilter, minFilter);
}


//-----------------------------------------------------------------------------
void Texture2D::createDepthTexture(int width, int height, GLint magFilter, GLint minFilter)
{
	_create(0, GL_DEPTH_COMPONENT16, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, magFilter, minFilter);
	bind(GL_TEXTURE0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2D::_create(GLvoid *data, GLint internalFormat, int width, int height, GLenum format, GLenum type, GLint magFilter, GLint minFilter)
{
	m_internalFormat = internalFormat;
	m_width = width;
	m_height = height;
	m_format = format;
	m_type = type;
	m_minFilter = magFilter;
	m_magFilter = minFilter;

	glGenTextures(1, &m_id);
	bind(GL_TEXTURE0);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2D::updateImage() {
	bind(GL_TEXTURE0);
	glTexImage2D(GL_TEXTURE_2D, 0, m_internalFormat, m_width, m_height, 0, m_format, m_type, m_image.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_minFilter);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2D::resize(int width, int height)
{
	m_width = width;
	m_height = height;
	bind(GL_TEXTURE0);
	glTexImage2D(GL_TEXTURE_2D, 0, m_internalFormat, width, height, 0, m_format, m_type, 0);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2D::bind(GLenum textureUnit)
{
	m_unit = textureUnit - GL_TEXTURE0;

	glActiveTexture(textureUnit);
	glBindTexture(GL_TEXTURE_2D, m_id);
	glActiveTexture(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2D::unbind(GLenum textureUnit)
{
	glActiveTexture(textureUnit);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::createColorTexture(int width, int height, GLsizei samples, GLint magFilter, GLint minFilter)
{
	_create(GL_RGBA8, width, height, samples, magFilter, minFilter);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::createDepthTexture(int width, int height, GLsizei samples, GLint magFilter, GLint minFilter)
{
	_create(GL_DEPTH_COMPONENT16, width, height, samples, magFilter, minFilter);
	bind(GL_TEXTURE0);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::_create(GLint internalFormat, int width, int height, GLsizei samples, GLint magFilter, GLint minFilter)
{
	m_internalFormat = internalFormat;
	m_width = width;
	m_height = height;
	m_minFilter = magFilter;
	m_magFilter = minFilter;
	m_samples = samples;

	glGenTextures(1, &m_id);
	bind(GL_TEXTURE0);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, internalFormat, width, height, GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MIN_FILTER, minFilter);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::updateImage() {
	bind(GL_TEXTURE0);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, m_samples, m_internalFormat, m_width, m_height, GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MAG_FILTER, m_magFilter);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MIN_FILTER, m_minFilter);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::resize(int width, int height)
{
	m_width = width;
	m_height = height;
	bind(GL_TEXTURE0);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, m_samples, m_internalFormat, width, height, true);
	unbind(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::bind(GLenum textureUnit)
{
	glActiveTexture(textureUnit);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_id);
	glActiveTexture(GL_TEXTURE0);
}


//-----------------------------------------------------------------------------
void Texture2DMultiSample::unbind(GLenum textureUnit)
{
	glActiveTexture(textureUnit);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	glActiveTexture(GL_TEXTURE0);
}
