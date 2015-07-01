/*
 * OpenCLPort.h
 *
 *  Created on: Jun 12, 2015
 *      Author: hedi7
 */

#ifndef SRC_CL_INTF_OPENCLPORT_H_
#define SRC_CL_INTF_OPENCLPORT_H_

#include "OpenCL.hpp"
#include <map>
// THis class is entriely responsible to limit the access to OpenCL kernels and objects
// Using too many OpenCL instances, will lead to a crash in the OpenCL engine.


class OpenCLPort {
public:
    static OpenCL& getInstance(const char *kernelname)
            {
                std::map<const char*,OpenCL>::iterator it = OpenCLPort::_openObj.find(kernelname);
//              std::cout << "Getinstance Called"<<std::endl;
                if(it != _openObj.end())
                {
                   //element found;
                   return it->second;
                }
                else{
//                  Init new element and put it into the list of known kernels

                    static OpenCL instance(kernelname); // Guaranteed to be destroyed.
                    std::pair<const char*,OpenCL> pair = std::make_pair(kernelname,instance);
                    OpenCLPort::_openObj.insert(OpenCLPort::_openObj.begin(),
                            pair);
                    return instance;
                }
            }

private:
//  DO not allow to create this object
    static std::map<const char*,OpenCL> _openObj;
    OpenCLPort(){};

    OpenCLPort(OpenCLPort const &other) = delete;
    OpenCLPort(OpenCLPort const &&other) = delete;
    void operator=(OpenCLPort const &other) = delete;

};


#endif /* SRC_CL_INTF_OPENCLPORT_H_ */
