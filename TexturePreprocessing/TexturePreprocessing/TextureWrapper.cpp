//
//  TextureWrapper.cpp
//  TexturePreprocessing
//
//  Created by gru on 12/14/15.
//  Copyright (c) 2015 Jan Gruber. All rights reserved.
//

#include "TextureWrapper.h"
#include <math.h>

OpenCLMgr* TextureWrapper::mgr = NULL;;

TextureWrapper::TextureWrapper(const std::string& filename)
{
    if(!filename.empty())
    {
        readRGBATiff(filename);
    }else
    {
        width = 0;
        imagelength = 0;
        inRaster = NULL;
        outRaster = NULL;
        
    }
}

TextureWrapper::~TextureWrapper()
{
    if(inRaster)
        delete [] inRaster;
    if(outRaster)
        delete [] outRaster;
}

void TextureWrapper::readRGBATiff(std::string filename)
{
    
    TIFF *tif=TIFFOpen(filename.c_str(), "r");
    if (tif) {
        TIFFRGBAImage img;
        char emsg[1024];
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
        if (TIFFRGBAImageBegin(&img, tif, 0, emsg)) {
            uint32 npixels;
            uint32* raster;
            
            npixels = img.width * img.height;
            raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
            if (raster != NULL) {
                if (TIFFRGBAImageGet(&img, raster, img.width, img.height)) {
                    std::cout<<"Read raster"<<std::endl;
                    inRaster = new uint32[npixels];
                    std::cout<<"Copy data raster"<<std::endl;
                    memcpy(inRaster, raster, npixels*sizeof(uint32));
                    
                }
                _TIFFfree(raster);
            }
            TIFFRGBAImageEnd(&img);
        }
        TIFFClose(tif);
    }
    
}

void TextureWrapper::writeRGBATiff(std::string filename)
{
    if(outRaster){
        TIFF *out= TIFFOpen(filename.c_str(), "w");
        int sampleperpixel = 4;
        TIFFSetField (out, TIFFTAG_IMAGEWIDTH, width); // sets the width of the image
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, imagelength); // sets the height of the image
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel); // sets number of channels per pixel
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8); // sets the size of the channels
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT); // sets the origin of the image.
        
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        
        uint8 *buf = NULL; // buffer used to store the row of pixel information for writing to file
        
        //Allocates memory to store the pixels of current row
        if (TIFFScanlineSize(out) == width)
            buf =(uint8 *)_TIFFmalloc(width);
        else
            buf = (uint8 *)_TIFFmalloc(TIFFScanlineSize(out));
        
        // Sets the strip size of the file to be size of one row of pixels
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width));
        
        //Writes image to the file one strip at a time
        for (uint32 row = 0; row < imagelength; row++)
        {
            memcpy(buf, &outRaster[(imagelength-row-1)*width], width*4);
            if (TIFFWriteScanline(out, buf, row, 0) < 0)
                break;
        }
        
        (void) TIFFClose(out);
        if (buf)
            _TIFFfree(buf);
        
    }
    
}


int TextureWrapper::preprocessTexture(uint16 flag)
{
    int status = 0;
    
    if(flag & PROCESS_AT_ONCE){
        status = blend();
    }
    
    if (flag & (PROCESS_X|PROCESS_Y)){
        status =  blendSeparate(flag);
    }
    
    if(flag & TEST){
        status = formTestMosaic();
    }
    
    return status;

}

int TextureWrapper::blend()
{
   outRaster = new uint32[imagelength*width];
    
    if(mgr->isValid()){
        cl_int status;
        size_t max_local_work_size;
        status = clGetKernelWorkGroupInfo(mgr->xyBlendKernel, NULL, CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),(void*)&max_local_work_size,NULL);
        size_t BLOCK_SIZE = sqrt(max_local_work_size);
        
        // create buffers
        cl_mem inBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, width*imagelength*sizeof(uint32), (void *)inRaster, &status);
        CHECK_SUCCESS("Error while creating input buffer!")
        
        
        cl_mem outBuffer = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, width*imagelength*sizeof(uint32), NULL, &status);
        CHECK_SUCCESS("Error while creating ouput buffer!")
        
        
        // Set kernel arguments
        int i=0;
        status |= clSetKernelArg(mgr->xyBlendKernel, i++, sizeof(cl_mem), (void *)&inBuffer);
        status |= clSetKernelArg(mgr->xyBlendKernel, i++, sizeof(cl_mem), (void *)&outBuffer);
        
        CHECK_SUCCESS("Error while setting kernel argument!")
        
        
        // Run the kernel.
        size_t global_work_size[2] = {width, imagelength};
        status = clEnqueueNDRangeKernel(mgr->commandQueue, mgr->xyBlendKernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        CHECK_SUCCESS("Error while enqueuing kernel!")
        
        status = clEnqueueReadBuffer(mgr->commandQueue, outBuffer, CL_TRUE, 0, width * imagelength*sizeof(uint32), outRaster, 0, NULL, NULL);
        CHECK_SUCCESS("Error while enqueuing read buffer!");
        
        // release buffers
        status = clReleaseMemObject(inBuffer);
        status = clReleaseMemObject(outBuffer);
        
        CHECK_SUCCESS("Error while releasing buffer!");
        
        return status;
        
    }
    return -1;

}

int TextureWrapper::blendSeparate(uint16 flag)
{
    outRaster = new uint32[imagelength*width];
    
    if(mgr->isValid()){
        cl_int status;
       
        
        // create buffers
        cl_mem inBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, width*imagelength*sizeof(uint32), (void *)inRaster, &status);
        CHECK_SUCCESS("Error while creating input buffer!")
        
        
        cl_mem outBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, width*imagelength*sizeof(uint32), NULL, &status);
        CHECK_SUCCESS("Error while creating ouput buffer!")
        size_t global_work_size[2] = {width, imagelength};

        if(flag & PROCESS_X){
            // Set kernel arguments
            int i=0;
            status |= clSetKernelArg(mgr->xBlendKernel, i++, sizeof(cl_mem), (void *)&inBuffer);
            status |= clSetKernelArg(mgr->xBlendKernel, i++, sizeof(cl_mem), (void *)&outBuffer);
            CHECK_SUCCESS("Error while setting kernel argument!")
            
            
            // Run the kernel
            status = clEnqueueNDRangeKernel(mgr->commandQueue, mgr->xBlendKernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
            CHECK_SUCCESS("Error while enqueuing kernel!")
            
            status = clEnqueueReadBuffer(mgr->commandQueue, outBuffer, CL_TRUE, 0, width * imagelength*sizeof(uint32), outRaster, 0, NULL, NULL);
            CHECK_SUCCESS("Error while enqueuing read buffer!");
        }
        
        if(flag & PROCESS_Y){
            //check, whether processing in X has already happened
            if(flag & PROCESS_X && (flag & PROCESS_Y)){
                status = clReleaseMemObject(inBuffer);
                inBuffer = outBuffer;
                outBuffer = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, width*imagelength*sizeof(uint32), NULL, &status);
                CHECK_SUCCESS("Error while creating ouput buffer!")
            }
            
            int i = 0;
            status |= clSetKernelArg(mgr->yBlendKernel, i++, sizeof(cl_mem), (void *)&inBuffer);
            status |= clSetKernelArg(mgr->yBlendKernel, i++, sizeof(cl_mem), (void *)&outBuffer);
            CHECK_SUCCESS("Error while setting kernel argument!")
            
            
            status = clEnqueueNDRangeKernel(mgr->commandQueue, mgr->yBlendKernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
            CHECK_SUCCESS("Error while enqueuing kernel!")
            
            status = clEnqueueReadBuffer(mgr->commandQueue, outBuffer, CL_TRUE, 0, width* imagelength*sizeof(uint32), outRaster, 0, NULL, NULL);
            CHECK_SUCCESS("Error while enqueuing read buffer!");
            
        }
        
        // release buffers
        status = clReleaseMemObject(inBuffer);
        status = clReleaseMemObject(outBuffer);
        
        CHECK_SUCCESS("Error while releasing buffer!");
        
        return status;
        
    }
    return -1;

}

int TextureWrapper::formTestMosaic()
{
    if(mgr->isValid()){
        cl_int status;
        cl_mem inBuffer;
        
        if(outRaster)
            // create buffer with outRaster data
            inBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, width*imagelength*sizeof(uint32), (void *)outRaster, &status);
        else{
            // outRaster is not existent, therefore use inRaster
            inBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, width*imagelength*sizeof(uint32), (void *)inRaster, &status);
            outRaster = new uint32[width*imagelength];
        }
        CHECK_SUCCESS("Error while creating input buffer!")
        
        
        cl_mem outBuffer = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, width*imagelength*sizeof(uint32), NULL, &status);
        CHECK_SUCCESS("Error while creating ouput buffer!")
        
        // Set kernel arguments
        int i=0;
        status |= clSetKernelArg(mgr->testMosaicKernel, i++, sizeof(cl_mem), (void *)&inBuffer);
        status |= clSetKernelArg(mgr->testMosaicKernel, i++, sizeof(cl_mem), (void *)&outBuffer);
        CHECK_SUCCESS("Error while setting kernel argument!")
        
        // Run the kernel.
        size_t global_work_size[2] = {width, imagelength};
        status = clEnqueueNDRangeKernel(mgr->commandQueue, mgr->testMosaicKernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        CHECK_SUCCESS("Error while enqueuing kernel!")
        
        status = clEnqueueReadBuffer(mgr->commandQueue, outBuffer, CL_TRUE, 0, width * imagelength*sizeof(uint32), outRaster, 0, NULL, NULL);
        CHECK_SUCCESS("Error while enqueuing read buffer!");
        
        // release buffers
        status = clReleaseMemObject(inBuffer);
        status = clReleaseMemObject(outBuffer);
        CHECK_SUCCESS("Error while releasing buffer!");
        
        return status;
        
    }
    return -1;
    
}


