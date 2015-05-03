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


using namespace std;

namespace util{

char* file_contents(const char* filepath);

}

#endif /* SRC_CL_INTF_UTIL_HPP_ */
