# Parallel-Computing-and-Algorithms-X033537-Project
Final project of course parallel computing and algorithms (X033537)


#Installation

##Ubuntu
First of all the packages for the dependencies need to be installed as:

```bash
sudo apt-get install cmake pkg-config python ocl-icd-dev ocl-icd-opencl-dev libdrm-dev libxfixes-dev libxext-dev llvm-3.5-dev clang-3.5 libclang-3.5-dev libtinfo-dev libedit-dev zlib1g-dev
```
Even though there exist a package in apt, this didnt work at all for my machine, so better build it by yourself, running:

```bash
git clone git://anongit.freedesktop.org/beignet
cd beignet
mkdir build
cd build
cmake ../
make
sudo make install
```

Finally, install a simple info tool for OpenCL:

```bash
sudo apt-get install clinfo
```
and run:
```
clinfo
```

Output should be showing something like:

```
Platform Name:				 Intel Gen OCL Driver
Number of devices:				 1
  Device Type:					 CL_DEVICE_TYPE_GPU
```
