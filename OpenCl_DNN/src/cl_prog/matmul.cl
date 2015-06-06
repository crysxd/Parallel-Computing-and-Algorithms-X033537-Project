#pragma OPENCL EXTENSION cl_intel_printf : enable
__kernel void mat_mul( __global const float* A,
                     __global const float* B,
                     const int wA,
                     const int wB,
                     __global float* C
                     )
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int tx = get_global_id(0);
   const int ty = get_global_id(1);
   float res = 0;

   for(unsigned i = 0; i < wA; ++i) {
      /**
       * We sine A and B are arrays, we skip over A every 'height' element
       * wA and over B we iterate columnwise, so we skip over the width
       *
       */
      res += A[ty*wA + i] * B[i*wB+tx];
   }
   C[ty*(wB) + tx] = res;
}
