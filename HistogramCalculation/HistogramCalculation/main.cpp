//
//  main.cpp
//  HistogramCalculation
//
//  Created by gru on 1/3/16.
//  Copyright (c) 2016 Jan Gruber. All rights reserved.
//

#include <iostream>
#include <string.h>

#include "tiffio.h"
#include <math.h>

#include <OpenCL/opencl.h>
#include"OpenCLMgr.h"
using namespace std;

#define BIN_COUNT 0x100
#define LOCAL_WORK_SIZE 0x20

//returns 32-bit RGBA
void readRGBATiff(const char* filename, uint32** inRaster, size_t* imagelength, size_t* width)
{
    TIFF *tif=TIFFOpen(filename, "r");
    if (tif) {
        TIFFRGBAImage img;
        char emsg[1024];
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, imagelength);
        if (TIFFRGBAImageBegin(&img, tif, 0, emsg)) {
            uint32 npixels;
            uint32* raster;
            
            npixels = img.width * img.height;
            raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
            if (raster != NULL) {
                if (TIFFRGBAImageGet(&img, raster, img.width, img.height)) {
                    std::cout<<"Read raster"<<std::endl;
                    (*inRaster) = new uint32[npixels];
                    std::cout<<"Copy data raster"<<std::endl;
                    memcpy(*inRaster, raster, npixels*sizeof(uint32));
                    
                }
                _TIFFfree(raster);
            }
            TIFFRGBAImageEnd(&img);
        }
        TIFFClose(tif);
    }
}

bool checkEquality ( int a[], cl_uint b[],int size, bool isVerbose)
{
    bool isEqual = true;
    for(int x=0; x<size; x++)
    {
        if(isVerbose)
            printf("A: %d, B: %d\n",a[x],b[x]);
        
        if(a[x] != b[x])
           isEqual=false;
    }
    return isEqual;
}

void calcHistogramOnCPU(uint32* img, size_t w, size_t h, int* histo){
    for (int i=0 ; i<0x100 ; i++)
        histo[i]=0;
   
    for (int i=0 ; i<w*h ; i++) {
        uint32 pix = img[i];
        
        unsigned char r = pix & 0xFF;
        unsigned char g = (pix >> 8) & 0xFF;
        unsigned char b = (pix >> 16) & 0xFF;
        float y = 0.299*r + 0.587*g + 0.114*b;

        histo[(unsigned char) y]+=1;
    }
}
int calcHistogramOnGPU(uint32* inRaster, size_t width, size_t imagelength, cl_uint *hist, size_t pixelCount, bool enableProfiling){
    OpenCLMgr mgr(enableProfiling);
    
    if(mgr.isValid()){
        cl_int status;
        
        // create buffers
        cl_mem inBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, width*imagelength*sizeof(uint32), (void *)inRaster, &status);
        CHECK_SUCCESS("Error while writing input buffer!")
        
        size_t len = width*imagelength;
        //cout<<"Number of pixels: "<< len<<endl;
        size_t pixPerWorkgroup = pixelCount*LOCAL_WORK_SIZE;
        size_t global_work_size[1] = {(len + pixPerWorkgroup-1)/pixPerWorkgroup*LOCAL_WORK_SIZE};//(len+8191)/8192*8192/256 = (len+8191)/8192*32
        //cout<<"Global work size: "<< global_work_size[0]<<endl;
        size_t local_work_size[1] = {LOCAL_WORK_SIZE};
        size_t num_workgroups = global_work_size[0]/LOCAL_WORK_SIZE;
        cout<<"Number of workgroups: "<< num_workgroups<<endl;
        
        
        cl_mem histBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE ,BIN_COUNT*num_workgroups*sizeof(cl_int), NULL, &status);
        CHECK_SUCCESS("Error while writing ouput buffer!")
        
        
        // Set kernel arguments
        int i=0;
        status |= clSetKernelArg(mgr.statKernel, i++, sizeof(cl_mem), (void *)&inBuffer);
        status |= clSetKernelArg(mgr.statKernel, i++, sizeof(size_t), &len);
        status |= clSetKernelArg(mgr.statKernel, i++, sizeof(cl_mem), (void *)&histBuffer);
        status |= clSetKernelArg(mgr.statKernel, i++, sizeof(size_t), &pixelCount);
        CHECK_SUCCESS("Error while setting kernel argument!")
        
        cl_event events[2];
        
        // Run the kernel.
        status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.statKernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[0]);
        CHECK_SUCCESS("Error while enqueuing kernel!")
        
        
        // Set kernel arguments
        i=0;
        status |= clSetKernelArg(mgr.reduceKernel, i++, sizeof(cl_mem), (void *)&histBuffer);
        status |= clSetKernelArg(mgr.reduceKernel, i++, sizeof(cl_int), &num_workgroups);
        global_work_size[0] = BIN_COUNT;
        local_work_size[0] = BIN_COUNT;
        
        
        status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.reduceKernel, 1, NULL, global_work_size, local_work_size, 1, &events[0], &events[1]);
        CHECK_SUCCESS("Error while enqueuing kernel!")
        
        status = clEnqueueReadBuffer(mgr.commandQueue, histBuffer, CL_TRUE, 0, BIN_COUNT*sizeof(cl_int), hist, 2, events, NULL);
        
        
        if(enableProfiling){
            clWaitForEvents(2 , events);
            cl_ulong startTime, endTime;
            status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL);
            status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, NULL);
            cl_ulong totalTime = endTime - startTime;
            cout<<"Duration of execution of regular kernel: "<<totalTime/pow(10,9)<<endl;
        }
        
        status = clReleaseMemObject(inBuffer);
        status = clReleaseMemObject(histBuffer);
        CHECK_SUCCESS("Error while releasing buffer!");
        
        status = clReleaseEvent(events[0]);
        status = clReleaseEvent(events[1]);
        CHECK_SUCCESS("Error while releasing events!");
        
        return status;
    }
    return -1;
}

int calcHistogramOnGPUAtomic(uint32* inRaster, size_t width, size_t imagelength, cl_uint *hist, size_t pixelCount, bool enableProfiling){
    OpenCLMgr mgr(enableProfiling);
    
    if(mgr.isValid()){
        cl_int status;
        
        // create buffers
        cl_mem inBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, width*imagelength*sizeof(uint32), (void *)inRaster, &status);
        CHECK_SUCCESS("Error while writing input buffer!")
        
        size_t len = width*imagelength;
        //cout<<"Number of pixels: "<< len<<endl;
        size_t pixPerWorkgroup = pixelCount*LOCAL_WORK_SIZE;
        size_t global_work_size[1] = {(len + pixPerWorkgroup-1)/pixPerWorkgroup*LOCAL_WORK_SIZE};//(len+8191)/8192*8192/256 = (len+8191)/8192*32
        //cout<<"Global work size: "<< global_work_size[0]<<endl;
        size_t local_work_size[1] = {LOCAL_WORK_SIZE};
        size_t num_workgroups = global_work_size[0]/LOCAL_WORK_SIZE;
        //cout<<"Number of workgroups: "<< num_workgroups<<endl;
        
        cl_mem histBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, BIN_COUNT*num_workgroups*sizeof(cl_int), NULL, &status);
        CHECK_SUCCESS("Error while writing ouput buffer!")
        
        
        // Set kernel arguments
        int i=0;
        status |= clSetKernelArg(mgr.statKernelAtomic, i++, sizeof(cl_mem), (void *)&inBuffer);
        status |= clSetKernelArg(mgr.statKernelAtomic, i++, sizeof(size_t), &len);
        status |= clSetKernelArg(mgr.statKernelAtomic, i++, sizeof(cl_mem), (void *)&histBuffer);
        status |= clSetKernelArg(mgr.statKernelAtomic, i++, sizeof(size_t), &pixelCount);
        CHECK_SUCCESS("Error while setting kernel argument!")
        
        cl_event events[2];
        
        // Run the kernel.
        status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.statKernelAtomic, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[0]);
        CHECK_SUCCESS("Error while enqueuing kernel!")
        
        
        // Set kernel arguments
        i=0;
        status |= clSetKernelArg(mgr.reduceKernel, i++, sizeof(cl_mem), (void *)&histBuffer);
        status |= clSetKernelArg(mgr.reduceKernel, i++, sizeof(cl_int), &num_workgroups);
        global_work_size[0] = BIN_COUNT;
        local_work_size[0] = BIN_COUNT;
        
        
        status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.reduceKernel, 1, NULL, global_work_size, local_work_size, 1, &events[0], &events[1]);
        CHECK_SUCCESS("Error while enqueuing kernel!")
        
        status = clEnqueueReadBuffer(mgr.commandQueue, histBuffer, CL_TRUE, 0, BIN_COUNT*sizeof(cl_int), hist, 2, events, NULL);
        
        
        if(enableProfiling){
            clWaitForEvents(2 , events);
            cl_ulong startTime, endTime;
            status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL);
            status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, NULL);
            cl_ulong totalTime = endTime - startTime;
            cout<<"Duration of execution of atomic kernel: "<<totalTime/pow(10,9)<<endl;
        }
        
        status = clReleaseMemObject(inBuffer);
        status = clReleaseMemObject(histBuffer);
        CHECK_SUCCESS("Error while releasing buffer!");
        
        status = clReleaseEvent(events[0]);
        status = clReleaseEvent(events[1]);
        CHECK_SUCCESS("Error while releasing events!");
        
        return status;
    }
    return -1;
}

int main(int argc, const char * argv[]) {
    uint32* inRaster = NULL;
    size_t imagelength;
    size_t width;
    
    std::string inputFile;
    std::cout << "Please enter input filepath: ";
    std::getline(std::cin, inputFile);
    
    bool enableProfiling=1;
    std::cout << "Enable profiling? (0/1) ";
    std::cin>>enableProfiling;
    
    readRGBATiff(inputFile.c_str(), &inRaster, &imagelength, &width);
    int histCPU[BIN_COUNT];

    calcHistogramOnCPU(inRaster, width, imagelength, histCPU);
    cl_uint* hist = new cl_uint[BIN_COUNT *sizeof(cl_uint)];
    
    
    if(enableProfiling)
        //best result around 0x1000 for regular, 0x100 for atomic
        for(size_t i=128; i<=0x2000;i=i<<1){
            cout<<i<<" pixels per workitem:"<<endl;
            cout<<"-------------------------------"<<endl;
            calcHistogramOnGPUAtomic(inRaster, width, imagelength, hist, i, enableProfiling);
            calcHistogramOnGPU(inRaster, width, imagelength, hist, i, enableProfiling);
            cout<<"==============================="<<endl;
        }
    else
        calcHistogramOnGPU(inRaster, width, imagelength, hist, 0x100, enableProfiling);
    
    bool result = checkEquality(histCPU, hist, BIN_COUNT, 0);

    if(result)
        printf("Results of CPU and GPU calculation are identical\n");
    else
        printf("Results of CPU and GPU calculation are not identical\n");
    return 0;
}
