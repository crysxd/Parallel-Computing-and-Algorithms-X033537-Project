# Parallel-Computing-and-Algorithms-X033537-Project
Final project of course parallel computing and algorithms (X033537)


#Installation

##Ubuntu
First of all the packages for the dependencies need to be installed as:

```bash
sudo apt-get install cmake pkg-config python ocl-icd-dev ocl-icd-opencl-dev libdrm-dev libxfixes-dev libxext-dev llvm-3.5-dev clang-3.5 libclang-3.5-dev libtinfo-dev libedit-dev zlib1g-dev
```

If you have a haskell CPU, you need to rebuild your kernel, by doing the following:

First get your current kernel from the official repositories.
```bash
git clone git://kernel.ubuntu.com/ubuntu/ubuntu-<release codename>.git
```
where <release codename> is trusty for 14.04, utopic for 14.10 etc.

Get the build tools:

```
sudo apt-get build-dep linux-image-$(uname -r)
```

After checking out cd into ubuntu-<release codename>

```
chmod a+x debian/scripts/*
chmod a+x debian/scripts/misc/*
fakeroot debian/rules clean
```

apply the [patch](https://01.org/zh/beignet/downloads/linux-kernel-patch-hsw-support).

Build the kernel by running:

```
fakeroot debian/rules clean
fakeroot debian/rules binary-headers binary-generic
```

After building, install:

```
cd ..
sudo dpkg -i linux*.deb
```

restart and make sure your GRUB has the new kernel as it's highest priority.


Even though there exist a package in apt, this didnt work at all for my machine, so better build it by yourself, running:

```bash
git clone git://anongit.freedesktop.org/beignet
cd beignet
git checkout Release_v1.0.0
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
