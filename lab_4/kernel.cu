#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <math.h>
#include <windows.h>
#include "kernel.h"

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 4096

using namespace std; //использовано пространство имен std


__global__ void addKernel(float a, float h, float *result, int N)
{
	//явное задание массивов в разделяемой памяти (кол-во ячеек = кол-ву нитей в блоке)
	__shared__ float result_shared[THREADS_PER_BLOCK];
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < N) {

		float x = a + h * i + h / 2;
		result_shared[threadIdx.x] = powf(logf(x), 2) / x * 7;
		__syncthreads();

		// редукция
		int j = blockDim.x / 2; // размер блока / 2

		while (j != 0) {
			if (threadIdx.x < j)
				result_shared[threadIdx.x] = result_shared[threadIdx.x] + result_shared[threadIdx.x + j];

			__syncthreads();
			j = j / 2;
		}

		//суммирование результатов каждого блока
		if (threadIdx.x == 0)
			atomicAdd(&(result[0]), result_shared[threadIdx.x]);
	}
}

// extern "C" - указывает, что данная функция будет будет доступна для другого программного компонента. 
//				имена объектов будут экспортироваться в совместимом с СИ виде.
// __declscpec(dllexport) - данная функция будет эскпортироваться из DLL
extern "C" __declspec(dllexport) void GpuIntegralCalc(float a, float b, long N, float *result, float *time)
{
	// вычислить шаг
	float h = (b - a) / N;

	float *dev_result;

	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 

	cudaMalloc((void**)&dev_result, sizeof(float));

	addKernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (a, h, dev_result, N);

	cudaMemcpy(result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop


	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop

	cout << endl;

	*time = gpuTime;
	*result = *result * h;
}
