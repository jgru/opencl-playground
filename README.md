# opencl-playground
This is a collection of some C/C++ code (within Xcode projects), which solves various tasks with the usage of the graphics adapter and OpenCL. I wrote those programs in the course of a parallel programming seminar to
* get familiar with OpenCL
* learn how to formulate problems in a way, that they can be processed efficiently on the GPU

Dependencies:
* OpenCL framework
* LibTiff

###Project description
---
Some further information to the separate projects: 

- TextureProcessing  
Some further information to the separate projects: 
- TextureProcessing
Reads the specified input image and creates a repeatable texture from it by performing a cyclic shift in x- and y-direction and blending of the shifted and original version of the image. 

- HistogramCalculation 
Calculates an 8-bit-histogram in an efficient way.

- PrefixSum 
Calculates the (exclusive) prefix sum of an array of integers.

