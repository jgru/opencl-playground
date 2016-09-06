//
//  main.cpp
//  image_processing
//
//  Created by gru on 11/30/15.
//  Copyright (c) 2015 Jan Gruber. All rights reserved.
//

#include <iostream>
#include <string.h>

#include <OpenCL/opencl.h>
#include "tiffio.h"

#include"OpenCLMgr.h"
#include "TextureWrapper.h"


int main(int argc, const char * argv[]) {

    
    //filenames are realtive to the directory ./build/products/
    std::string inputFile = "./test_img_1.tif";
    std::string outputFile = "./new.tif";

    OpenCLMgr mgr;
    TextureWrapper::mgr = &mgr;
    
    //Inits TextureWrapper
    TextureWrapper texWrapper(inputFile.c_str());
    
    //Processes the image
    texWrapper.preprocessTexture(TextureWrapper::PROCESS_AT_ONCE | TextureWrapper::TEST);
    //texWrapper.preprocessTexture(TextureWrapper::PROCESS_X | TextureWrapper::PROCESS_Y | TextureWrapper::TEST);
    
    //Writes to disk
    texWrapper.writeRGBATiff(outputFile.c_str());
    
    
    return 0;
}
