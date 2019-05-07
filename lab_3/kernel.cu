
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

__global__ void addKernel(float a, float h, float *result, float *buf, int N)
{
	//явное задание массивов в разделяемой памяти (кол-во ячеек = кол-ву нитей в блоке)
	__shared__ float result_shared[THREADS_PER_BLOCK];
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < N) {

		float x = a + h * i + h / 2;

		if (x <= 10) {

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
}

int main()
{
	setlocale(LC_ALL, "Russian");

	float a = 1;
	float b = 10;
	int N = THREADS_PER_BLOCK * BLOCKS_PER_GRID;

	// вычислить шаг
	float h = (b - a) / N;
	cout << "шаг: " << h << endl;

	cpuIntegralCalc(a, b, h, N);
	gpuIntegralCalc(a, b, h, N);

	return 0;
}

void cpuIntegralCalc(double a, double b, double h, int N)
{
	double result = 0;

	LARGE_INTEGER timerFrequency, timerStart, timerStop;
	QueryPerformanceFrequency(&timerFrequency);
	QueryPerformanceCounter(&timerStart);

	for (double x = a; x < b; x = x + h)
	{
		double temp_x = x + (h / 2);
		result += pow(log(temp_x), 2) / temp_x * 7;
	}

	QueryPerformanceCounter(&timerStop);

	//Выводим время выполнения на CPU (в мс) 
	double cpuTime = (((double)timerStop.QuadPart - (double)timerStart.QuadPart) / (double)timerFrequency.QuadPart) * 1000;

	cout << "Последовательный вариант: время работы в мс: " << cpuTime << endl;
	cout << "Ответ: " << result * h << endl << endl;
}

void gpuIntegralCalc(double a, double b, float h, int N)
{
	float *host_result;
	float *dev_result;
	float *dev_buf;

	host_result = (float*)malloc(sizeof(float));

	cudaMalloc((void**)&dev_result, sizeof(float));
	cudaMalloc((void**)&dev_buf, N * sizeof(float));


	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 


	addKernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (a, h, dev_result, dev_buf, N);

	// копирование данных с девайса на хост
	cudaMemcpy(host_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);


	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop


	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop

	cout << "Время работы алгоритма на gpu, в мс: " << gpuTime << endl; // Печатаем время
	cout << "Ответ: " << *host_result * h << endl;

	free(host_result);
	cudaFree(dev_result);
	cudaFree(dev_buf);
}
