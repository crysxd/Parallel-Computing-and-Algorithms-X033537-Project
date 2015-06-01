#pragma OPENCL EXTENSION cl_intel_printf : enable


float helper(){

    printf("Helping");
    return 0.0;
}

__kernel void sigmoid ()
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
   const int idx = get_global_id(0);
   /* Now each work-item asks itself: "is my ID inside the vector's range?"
   If the answer is YES, the work-item performs the corresponding computation*/
   // if (idx < num)
   res[idx] = src_a[idx] + src_b[idx];
   helper();
   printf("IN the GPU: %f %f %f\n",src_a[idx], src_b[idx],res[idx]);
}
