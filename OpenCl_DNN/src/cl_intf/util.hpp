/*
 * util.hpp
 *
 *  Created on: May 3, 2015
 *      Author: hedi7
 */

#ifndef SRC_CL_INTF_UTIL_HPP_
#define SRC_CL_INTF_UTIL_HPP_
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <CL/cl.hpp>
#include <random>


using namespace std;

namespace util{

char* file_contents(const char* filepath);

template <typename T>
void randinit(int min, int max, std::vector<T> arr);
float randfloat(int min, int max);
struct GPU_Buffer{
    cl::Buffer buffer;
    int datalength;
};

}


#endif /* SRC_CL_INTF_UTIL_HPP_ */
