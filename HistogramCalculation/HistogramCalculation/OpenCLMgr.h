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
	OpenCLMgr(bool enableProfiling);
	~OpenCLMgr();

	int isValid() {return valid;}
    bool enableProfiling;
    cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	
	cl_kernel statKernel;
    cl_kernel statKernelAtomic;
    cl_kernel reduceKernel;


private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};