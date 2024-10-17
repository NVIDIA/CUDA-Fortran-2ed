extern "C" __global__ void Ckernel(float *a, float b)
{
  a[threadIdx.x] = b;
}

extern "C" __device__ float Cdevicefun(float a)
{
  return 2*a;
}
