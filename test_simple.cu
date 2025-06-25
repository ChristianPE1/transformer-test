#include <iostream>
#include <cuda_runtime.h>
#include "include/common.cuh"

__global__ void testKernel()
{
   printf("CUDA está funcionando! Thread ID: %d\n", threadIdx.x);
}

int main()
{
   std::cout << "=== Prueba Simple CUDA ===" << std::endl;

   // Verificar dispositivos CUDA
   int deviceCount;
   cudaGetDeviceCount(&deviceCount);
   std::cout << "Dispositivos CUDA encontrados: " << deviceCount << std::endl;

   if (deviceCount > 0)
   {
      // Lanzar kernel simple
      testKernel<<<1, 5>>>();
      cudaDeviceSynchronize();

      std::cout << "Prueba CUDA completada exitosamente!" << std::endl;
   }
   else
   {
      std::cout << "No se encontraron dispositivos CUDA" << std::endl;
   }

   return 0;
}