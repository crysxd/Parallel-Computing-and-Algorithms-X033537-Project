#pragma OPENCL EXTENSION cl_intel_printf : enable

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


// __kernel void mat_mul2 ( __global const float* A,
//                      __global const float* B,
//                      const int wA,
//                      const int wB,
//                      __global float* C){

//    int gx = get_global_id(0);
//    int gy = get_global_id(1);

//    int tx = get_local_id(0);
//    int ty = get_local_id(1);

//    const int blocksizex = get_local_size(0);
//    const int blocksizey = get_local_size(1);

//    int blocksize = blocksizex* blocksizey;

//    int aBegin = wA * blocksize * gy;
//    int aEnd = aBegin + wA - 1;
//    int astep = blocksize;


//    int bBegin = gx * blocksize;
//    int bstep = wB * blocksize;

//    float cres = 0.0;

//    for (int a = aBegin,b=bBegin; a <= aEnd; a+= astep, b+=bstep)
//    {
//       __local float Atmp[blocksizex][blocksizey];
//       __local float Btmp[blocksizex][blocksizey];

//       Atmp[ty][tx] = A[a+wA * ty + tx];
//       Btmp[ty][tx] = B[b+wB * ty + tx];

//       barrier(CLK_LOCAL_MEM_FENCE);
//    }


// // Get the submatrices for A


// }

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
      res += A[tx*wA + i] * B[i*wB+ty];
   }
   // printf("wA %i wB %i tx %i ty %i to %i value %.2f \n",wA,wB,tx,ty,tx*(wB) + ty,res);
   // if(res < 9000){
   // printf(" Result of tx %i ty %i is %.1f",tx,ty,res);
   // }
   C[tx*(wB) + ty] = res;
}



__kernel void sigmoid(const int wSrc, __global const float* src, __global float* output)
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int idx = get_global_id(0);
   const int idy = get_global_id(1);

   /* Now each work-item asks itself: "is my ID inside the vector's range?"
   If the answer is YES, the work-item performs the corresponding computation*/
   // if (idx < num)
   for (int i = 0; i < 100000; i++) {
   	output[idy*wSrc+idx] = 1.f/ (1.f + exp(-src[idy*wSrc+idx]));
   }
}


__kernel void cl_tanh(const int wSrc,__global const float* src, __global float* output)
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int idx = get_global_id(0);
   const int idy = get_global_id(1);

   /* Now each work-item asks itself: "is my ID inside the vector's range?"
   If the answer is YES, the work-item performs the corresponding computation*/
   // if (idx < num)
   output[idy*wSrc +idx] = tanh(src[idy*wSrc +idx]);
}

__kernel void sigmoidgrad(const int wSrc, __global const float* src, __global float* output)
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int idx = get_global_id(0);
   const int idy = get_global_id(1);

   // Compute elementwise s(x)(1-s(x))
   output[idy*wSrc+idx] = 1.f/ (1.f + exp(-src[idy*wSrc+idx])) * (1.f - 1.f/ (1.f + exp(-src[idy*wSrc+idx])));
}
__kernel void mult(const int wSrc, __global const float* A,__global const float* B,__global float* output)
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int idx = get_global_id(0);
   const int idy = get_global_id(1);

   // Compute elementwise s(x)(1-s(x))
   output[idy*wSrc+idx] = A[idy*wSrc+idx] * B[idy*wSrc+idx];
}
