
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

#include "OpenCLMgr.h"


OpenCLMgr::OpenCLMgr(bool enableProfiling)
{
    this->enableProfiling = enableProfiling;
	context = 0;
	commandQueue = 0;
	program = 0;
	statKernel = 0;
    statKernelAtomic = 0;
    reduceKernel = 0;
	valid = (init()==SUCCESS);
}


OpenCLMgr::~OpenCLMgr()
{
	cl_int status;

	//Release kernels
	if (statKernel) status = clReleaseKernel(statKernel);
   	if (statKernelAtomic) status = clReleaseKernel(statKernelAtomic);
    if (reduceKernel) status = clReleaseKernel(reduceKernel);

    //Release the program object.
	if (program) status = clReleaseProgram(program);
    
	//Release  Command queue.
	if (commandQueue) status = clReleaseCommandQueue(commandQueue);	    

	//Release context.
	if (context) 	  status = clReleaseContext(context);				
}


/* convert the kernel file into a string */
int OpenCLMgr::convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size+1];
		if(!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout<<"Error: failed to open file\n:"<<filename<<endl;
	return FAILURE;
}


cl_int OpenCLMgr::init()
{
	cl_uint deviceNo = 1;

	// Getting platforms and choose an available one.
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	CHECK_SUCCESS("Error: Getting platforms!")

	// For clarity, choose the first available platform.
	if(numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
		CHECK_SUCCESS("Error: Getting platforms ids")
	}

	// Query devices and choose a GPU device if has one. Otherwise use the CPU as device.*/
	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);	
	CHECK_SUCCESS("Error: Getting device ids")
	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		CHECK_SUCCESS("Error: Getting number of cpu devices")
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		CHECK_SUCCESS("Error: Getting cpu device id")
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		CHECK_SUCCESS("Error: Getting gpu device id")
	}
	
	if (deviceNo>=numDevices)
		deviceNo=0;
    device = *(devices+deviceNo);
    
	// Create context
	context = clCreateContext(NULL,1, devices+deviceNo,NULL,NULL,NULL);
	CHECK_SUCCESS("Error: creating OpenCL context")

    // Creating command queue associate with the context
    
    if(enableProfiling){
        commandQueue = clCreateCommandQueue(context, devices[deviceNo], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE, &status);
        
        /*this is neccessary to work around some strange behaviour of Apple's OpenCL implementetation,
         *which throws an error (-30), if one tries to set the  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
         *flag in clCreateCommandQueue(...). Furtheron it is not possible to specify CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE. 
         Then profiling doesn't work, it throws then (-7)
         */
        #ifdef __APPLE__
                commandQueue = clCreateCommandQueue(context, devices[deviceNo], CL_QUEUE_PROFILING_ENABLE, &status);
        #endif
    }else{
        commandQueue = clCreateCommandQueue(context, devices[deviceNo], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
        
        #ifdef __APPLE__
            cl_queue_properties_APPLE props;
            props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            commandQueue = clCreateCommandQueueWithPropertiesAPPLE(context,device,&props,&status);
        
        #endif
        
    }

    CHECK_SUCCESS("Error: creating command queue")
	// Create program object 
	const char *filename = "./image_stats.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	CHECK_SUCCESS("Error: loading OpenCL file")

	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};
	program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
	CHECK_SUCCESS("Error: creating OpenCL program")
	
	// Build program. 
    status=clBuildProgram(program, 1,devices+deviceNo,NULL,NULL,NULL);
    if (status) {
        char msg[120000];
        clGetProgramBuildInfo(program, devices[deviceNo], CL_PROGRAM_BUILD_LOG, sizeof(msg), msg, NULL);
        cerr << "=== build failed ===\n" << msg << endl;
        getc(stdin);
        return FAILURE;
    }

	// Create kernel objects 
	statKernel = clCreateKernel(program, "calcStatistic", &status);
	CHECK_SUCCESS("Error: creating calcStatistic-Kernel")
    statKernelAtomic = clCreateKernel(program, "calcStatisticAtomic", &status);
    CHECK_SUCCESS("Error: creating calcStatisticAtomic-Kernel")
    reduceKernel = clCreateKernel(program, "reduceStatistic", &status);
    CHECK_SUCCESS("Error: creating calcStatistic-Kernel")


	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}


	return status;
}

