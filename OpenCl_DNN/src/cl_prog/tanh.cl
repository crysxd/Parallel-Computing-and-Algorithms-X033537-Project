#ifndef defined cl_intel_printf
    #pragma OPENCL EXTENSION cl_intel_printf : enable
#endif
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void tanh(__global const float* src, __global float* output)
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int idx = get_global_id(0);

   /* Now each work-item asks itself: "is my ID inside the vector's range?"
   If the answer is YES, the work-item performs the corresponding computation*/
   // if (idx < num)
   output[idx] = tanh(src[idx]);
}
