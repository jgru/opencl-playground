//
//  main.cpp
//  PrefixSum
//
//  Created by gru on 12/7/15.
//  Copyright (c) 2015 Jan Gruber. All rights reserved.
//

#include <iostream>
#include "OpenCLMgr.h"
#include "ArrayUtils.h"

using namespace std;

//helper funciton to create and fill C-style array
cl_int* createArray(const size_t& arraySize) {
    cl_int* arr = new cl_int[arraySize];
    for (int i=0; i<arraySize; i++) {
        arr[i]=i;
    }
    return arr;
}


int main(int argc, const char * argv[]) {

    //array creation
    size_t len=80000005;
    cl_int* arr = createArray(len);
    cl_int* prefix_sum_a = new cl_int[len];
    cl_int* prefix_sum_b = new cl_int[len];
    
    //calculate prefix sum on gpu & cpu
    int status = ArrayUtils::calcPrefixSumOnGPU(arr, prefix_sum_a, len);
    ArrayUtils::calcPrefixSumOnCPU(arr, prefix_sum_b, len);
  
    //print result
    if(status==CL_SUCCESS){
        if(std::equal(prefix_sum_a, prefix_sum_a+len, prefix_sum_b))
            cout<<"CPU calculation an GPU calculation are equal"<<endl;
        else
            cout<<"CPU calculation an GPU calculation are NOT equal"<<endl;
    }
    
    for(int i=0; i<len;i++){
        if(prefix_sum_a[i]!=prefix_sum_b[i]){
            cout<<i<<": "<<prefix_sum_a[i]<<"-"<<prefix_sum_b[i]<<endl;
           break;
        }else if(i==(len-1))
            cout<<i<<": "<<prefix_sum_a[i]<<"-"<<prefix_sum_b[i]<<endl;
    }
    cout<<len-1<<": "<<prefix_sum_a[len-1]<<"-"<<prefix_sum_b[len-1]<<endl;

    //clean up
    delete[] arr;
    delete[] prefix_sum_a;
    delete[] prefix_sum_b;

    return status;
}
