#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <math.h>
#include <windows.h>
#include "kernel.h"

#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 2048

using namespace std; //использовано пространство имен std

__global__ void gaussKernel(float *Im, int n)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;

	//blockDim.x*gridDim.x - нитей в строке
	int shift = x + y * blockDim.x*gridDim.x;

	if (shift < n)
	{
		float spot1 = expf(-(powf((y - 344.0) / 512, 2) + powf((x - 42.0) / 512, 2)) / 0.0055);
		float spot2 = expf(-(powf((y - 244.0) / 512, 2) + powf((x - 342.0) / 512, 2)) / 0.0055);
		float spot3 = expf(-(powf((y - 44.0) / 512, 2) + powf((x - 342.0) / 512, 2)) / 0.0055);

		float val = 255 * (spot1 + spot2 + spot3);

		val = val > 255 ? 255 : val;

		Im[shift] = val;
		__syncthreads();
	}
}


__global__ void backProjectionKernel(float* device_XY, float* device_discretP, int host_NN, int host_M, float host_dQ) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int shift = x + y * blockDim.x*gridDim.x;
	float Q = host_dQ; // Начальный угол равен угловому шагу

	if (shift < host_NN*host_NN) {// Чтобы не выйти за диапазон массива

		for (int i = 0; i < (host_M - 1); i++) {// Цикл по “M” проекциям

			float v = (x - host_NN / 2.f)*cosf(3.14f / 180.f*Q) + (y - host_NN / 2.f)*sinf(3.14f / 180.f*Q);
			int r = (int)(v + host_NN / 2.f); // Приводим к целому типу

			if ((r < (host_NN - 1)) && (r > -1.f)) {// Чтобы не выйти за диапазон своей проекции
				device_XY[shift] = device_XY[shift] + device_discretP[r + i * host_NN];

				// Возможно использование:
				// atomicAdd(&(device_XY[shift]), device_discretP[r+i*host_NN]); }
				__syncthreads(); // Синхронизация
				Q = Q + host_dQ; // Увеличиваем угол

			}
		}
	}
}


extern "C" __declspec(dllexport) void GpuGaussCalc(float *hostIm, float *time)
{
	// вычислить шаг
	float *dev_result;
	int n = 512 * 512;

	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 

	cudaMalloc((void**)&dev_result, n * sizeof(float));

	dim3 blocks(16, 16);
	dim3 threads(32, 32);

	gaussKernel << <blocks, threads >> > (dev_result, n);

	cudaMemcpy(hostIm, dev_result, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop

	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop

	*time = gpuTime;

	cudaFree(dev_result);
}

extern "C" __declspec(dllexport) float** BackProjection(float *time) {

	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 

	dim3 blocks(16, 16);
	dim3 threads(32, 32);

	backProjectionKernel << <blocks, threads >> > (dev_result, n);

	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop

	*time = gpuTime;

	float **test = 0;

	return test;

}