#pragma once


#include <OpenCL/opencl.h>
#include <string>
#include <iostream>
#include <string>

#define SUCCESS 0
#define FAILURE 1
#define BLOCK_SIZE 16


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
	
	cl_kernel prefixSumKernel;
    cl_kernel mergeKernel;

private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};