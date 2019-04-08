#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include <map>
#include <vector>
#include <stdexcept>


class GLSLProgramException : public std::runtime_error {
public:
    GLSLProgramException(const std::string &msg) :
            std::runtime_error(msg) {}
};

namespace GLSLShader {
    enum GLSLShaderType {
        VERTEX = GL_VERTEX_SHADER,
        FRAGMENT = GL_FRAGMENT_SHADER,
        GEOMETRY = GL_GEOMETRY_SHADER,
        TESS_CONTROL = GL_TESS_CONTROL_SHADER,
        TESS_EVALUATION = GL_TESS_EVALUATION_SHADER,
        COMPUTE = GL_COMPUTE_SHADER
    };
};

class GLSLProgram {
private:
	HGLRC hglrc;
    GLuint handle;
	GLuint vaoHandle;
    bool linked;
    std::map<std::string, int> uniformLocations;

    GLint getUniformLocation(const char *name);

    bool fileExists(const std::string &fileName);

    std::string getExtension(const char *fileName);

    // Make these private in order to make the object non-copyable
    GLSLProgram(const GLSLProgram &other) {}

    GLSLProgram &operator=(const GLSLProgram &other) { return *this; }

public:
    GLSLProgram();

    ~GLSLProgram();

	void createContex(HWND hWnd);

    void compileShader(const char *fileName);

    void compileShader(const char *fileName, GLSLShader::GLSLShaderType type);

    void compileShader(const std::string &source, GLSLShader::GLSLShaderType type,
                       const char *fileName = NULL);

    void link();

    void validate();

    void use();

	void draw(GLenum mode, GLint first, GLsizei count);

    int getHandle();

    bool isLinked();

    void bindAttribLocation(GLuint location, const char *name);

    void bindFragDataLocation(GLuint location, const char *name);

	void setAtribute(GLuint location, const std::vector<float> &data, GLint size);

    void setUniform(const char *name, float x, float y, float z);

    void setUniform(const char *name, const glm::vec2 &v);

    void setUniform(const char *name, const glm::vec3 &v);

    void setUniform(const char *name, const glm::vec4 &v);

    void setUniform(const char *name, const glm::mat4 &m);

    void setUniform(const char *name, const glm::mat3 &m);

    void setUniform(const char *name, float val);

    void setUniform(const char *name, int val);

    void setUniform(const char *name, bool val);

    void setUniform(const char *name, GLuint val);

    void findUniformLocations();

    void printActiveUniforms();

    void printActiveUniformBlocks();

    void printActiveAttribs();

	void PrintGPUVersion(bool display_extensions = false);

    const char *getTypeString(GLenum type);
};
