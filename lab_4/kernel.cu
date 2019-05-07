
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <math.h>
#include <windows.h>

//#define THREADS_PER_BLOCK 128
//#define BLOCKS_PER_GRID 4096

using namespace std; //использовано пространство имен std

__global__ void addKernel(float a, float h, float *result, float *buf, int N)
{
	//явное задание массивов в разделяемой памяти (кол-во ячеек = кол-ву нитей в блоке)
	__shared__ float result_shared[128];
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

		//запись результата из разделяемой памяти обратно в глобальную
		buf[threadIdx.x + blockDim.x*blockIdx.x] = result_shared[threadIdx.x];
		__syncthreads();

		//суммирование результатов каждого блока
		if (threadIdx.x == 0)
			atomicAdd(result, buf[blockDim.x*blockIdx.x]);
	}
}

extern "C" __declspec(dllexport) void GpuIntegralCalc(float *a, float *b, int *N, float *test, float *time)
{
	// вычислить шаг
	float a1 = 1;
	float b2 = 10;
	int N1 = 4096 * 128;


	float h = (b2 - a1) / N1;
	cout << "шаг: " << h << endl;

	float *host_result;
	float *dev_result;
	float *dev_buf;
	float *host_buf;

	host_result = (float*)malloc(sizeof(float));
	host_buf = (float*)malloc(N1 * sizeof(float));

	cudaMalloc((void**)&dev_result, sizeof(float));
	cudaMalloc((void**)&dev_result, sizeof(float));
	cudaMalloc((void**)&dev_result, sizeof(float));
	cudaMalloc((void**)&dev_result, sizeof(float));
	cudaMalloc((void**)&dev_buf, N1 * sizeof(float));


	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 


	addKernel << <4096, 128 >> > (a1, h, dev_result, dev_buf, N1);

	// копирование данных с девайса на хост
	cudaMemcpy(host_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_buf, dev_buf, N1 * sizeof(float), cudaMemcpyDeviceToHost);


	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop


	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop

	for (size_t i = 0; i < 100; i++)
	{
		cout << host_buf[i] << endl;
	}

	cout << endl;

	cout << gpuTime << endl << *host_result << endl;

	*time = gpuTime;
	*test = *host_result;

	cudaFree(dev_result);
	cudaFree(dev_buf);
}