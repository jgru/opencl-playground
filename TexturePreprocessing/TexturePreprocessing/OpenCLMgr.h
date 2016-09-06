#pragma once


#include <string>
#include <iostream>
#include <string>

#include <OpenCL/opencl.h>

#define SUCCESS 0
#define FAILURE 1

#define CHECK_SUCCESS(msg) \
		if (status!=SUCCESS) { \
            std::cout << msg << std::endl; \
			return FAILURE; \
		}


class OpenCLMgr
{
public:
	OpenCLMgr();
	~OpenCLMgr();

	int isValid() {return valid;}

	
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	
	cl_kernel xBlendKernel;
    cl_kernel yBlendKernel;
    cl_kernel xyBlendKernel;
    cl_kernel testMosaicKernel;


private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};