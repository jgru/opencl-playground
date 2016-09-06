//
//  ArrayUtils.cpp
//  PrefixSum
//
//  Created by gru on 1/6/16.
//  Copyright (c) 2016 Jan Gruber. All rights reserved.
//

#include "ArrayUtils.h"
#include<OpenCL/OpenCL.h>
#include"OpenCLMgr.h"

using namespace std;

#define lws 256

size_t ArrayUtils::calcNextMultiple(size_t len, size_t n)
{
    return (((len-1)/n+1)*n);
}


int ArrayUtils::calcPrefixSumOnGPU(cl_int *input, cl_int *output, size_t len)
{
    OpenCLMgr mgr;
    
    if(mgr.isValid()){
        std::cout<<"Summing on GPU"<<std::endl;
        std::cout<<"----------------------"<<std::endl;
        
        size_t local_size=256;
        size_t global_len= calcNextMultiple(len,local_size);
        std::cout<<"Input length: "<<len<<std::endl;
        std::cout<<"Local worksize:"<<local_size<<std::endl;
        std::cout<<"Global_len: "<<global_len<<std::endl;
        
        cl_int status= 0;
        cl_mem aBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, len * sizeof(cl_int),(void *) input, &status);
        CHECK_SUCCESS("Error while creating cl_mem buffer")
        
        cl_mem bBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, global_len * sizeof(cl_int), NULL, &status);
        CHECK_SUCCESS("Error while creating cl_mem buffer")
        
        prefixSumRecursive(mgr, aBuffer, bBuffer, global_len, lws);
        
        status = clEnqueueReadBuffer(mgr.commandQueue, bBuffer, CL_TRUE, 0, len * sizeof(cl_int), output, 0, NULL, NULL);
        CHECK_SUCCESS("Error reading buffer")

        
        //dealloc buffers
        
        return status;
    }
    return -1;

}

int ArrayUtils::prefixSumRecursive(OpenCLMgr& mgr, cl_mem aBuffer, cl_mem bBuffer, size_t global_len, size_t local_size)
{
    cout<<"ArrayUtils::prefixSumRecursive(...)"<<endl;
    cout<<"Global len: "<<global_len<<endl;
    cl_int status=0;


    size_t resulting_length = calcNextMultiple((global_len/local_size), local_size);
    cl_mem cBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE , resulting_length * sizeof(cl_int), NULL, &status);
    cl_mem dBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE , resulting_length * sizeof(cl_int), NULL, &status);
    CHECK_SUCCESS("Error creating buffers")
    
    int i = 0;
    status |= clSetKernelArg( mgr.prefixSumKernel, i++, sizeof(cl_mem), &aBuffer);
    status |= clSetKernelArg( mgr.prefixSumKernel, i++, sizeof(cl_mem), &bBuffer);
    status |= clSetKernelArg( mgr.prefixSumKernel, i++, sizeof(cl_mem), &cBuffer);
    status |= clSetKernelArg( mgr.prefixSumKernel, i++, local_size *sizeof(cl_int), NULL );//NULL tells the kernel that its third argument can be allocated from global memory or from local memory.
    CHECK_SUCCESS("Error setting kernel args")
    
    size_t global_work_size[1] = {global_len};
    size_t local_work_size[1] = {local_size};
    
    status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.prefixSumKernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_SUCCESS("Error enqueuing kernel")
    
    if((global_len/local_size)>1){
        status = prefixSumRecursive(mgr, cBuffer, dBuffer, resulting_length, local_size);
        
        //final summing
        i = 0;
        status = clSetKernelArg( mgr.mergeKernel, i++, sizeof(cl_mem), &dBuffer );
        status =  clSetKernelArg( mgr.mergeKernel, i++, sizeof(cl_mem), &bBuffer );
        CHECK_SUCCESS("Error setting kernel args")
        
        global_work_size[0] = global_len;
        status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.mergeKernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        CHECK_SUCCESS("Error enqueuing merging kernel")
    }
    //dealloc
    
    
    return status;
}



void ArrayUtils::calcPrefixSumOnCPU(cl_int *input, cl_int *prefix_sum, size_t size)
{
    prefix_sum[0] = 0;
    for (int i=1 ; i<size ; i++){
            prefix_sum[i] = prefix_sum[i-1]+input[i-1];
    }
}