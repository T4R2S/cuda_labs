
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <math.h>
#include <windows.h>
#define BLOCKS_PER_GRID 512 //число блоков в сетке

using namespace std; //использовано пространство имен std

__global__ void myKernel(float *A, float *B, float *C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) {
		float x = 3.14f / 180.0f*i;
		C[i] = (A[i] * powf(cosf(x), 4)) + (B[i] * expf(i / N));
	}
}

void myCpuFunction(float *A, float *B, float *C, int N) {

	for (size_t i = 0; i < N; i++)
	{
		float x = 3.14f / 180.0f*i;
		C[i] = (A[i] * powf(cosf(x), 4)) + (B[i] * expf(i / N));
	}
}

void startCPUCode_CountRuntime(cudaDeviceProp devProp) {

	float *hostA, *hostB, *hostC;
	int N = devProp.maxThreadsPerBlock * BLOCKS_PER_GRID;

	hostA = (float*)malloc(N * sizeof(float));
	hostB = (float*)malloc(N * sizeof(float));
	hostC = (float*)malloc(N * sizeof(float));

	//Инициализация массивов случайными значениями 
	for (int i = 0; i < N; i++)
	{
		hostA[i] = rand();
		hostB[i] = rand();
	}

	LARGE_INTEGER timerFrequency, timerStart, timerStop;
	QueryPerformanceFrequency(&timerFrequency);
	QueryPerformanceCounter(&timerStart);

	myCpuFunction(hostA, hostB, hostC, N);

	QueryPerformanceCounter(&timerStop);

	//Выводим время выполнения на CPU (в мс) 
	double cpuTime = (((double)timerStop.QuadPart - (double)timerStart.QuadPart) / (double)timerFrequency.QuadPart) * 1000;
	cout << "Последовательный вариант: время работы в мс: " << cpuTime << endl;

	free(hostA);
	free(hostB);
	free(hostC);
}

void startKernel_CountRuntime(cudaDeviceProp devProp, dim3 threadsPerBlock, dim3 blocksPerGrid) {

	float *hostA, *hostB, *hostC;
	int N = devProp.maxThreadsPerBlock * BLOCKS_PER_GRID;

	hostA = (float*)malloc(N * sizeof(float));
	hostB = (float*)malloc(N * sizeof(float));
	hostC = (float*)malloc(N * sizeof(float));

	float *deviceA, *deviceB, *deviceC;

	cudaMalloc((void**)&deviceA, N * sizeof(float));
	cudaMalloc((void**)&deviceB, N * sizeof(float));
	cudaMalloc((void**)&deviceC, N * sizeof(float));

	//Инициализация массивов случайными значениями
	for (int i = 0; i < N; i++)
	{
		hostA[i] = rand();
		hostB[i] = rand();
	}

	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaMemcpy(deviceA, hostA, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 

	myKernel << <blocksPerGrid, threadsPerBlock >> > (deviceA, deviceB, deviceC, N);

	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop

	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop

	cudaMemcpy(hostC, deviceC, N * sizeof(float), cudaMemcpyDeviceToHost);

	cout << "\t" << gpuTime; // Печатаем время

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);
}

void startKernel_CountRuntimeAndDeviceToHostCopyTime(cudaDeviceProp devProp, dim3 threadsPerBlock, dim3 blocksPerGrid) {
	
	float *hostA, *hostB, *hostC;
	int N = devProp.maxThreadsPerBlock * BLOCKS_PER_GRID;

	hostA = (float*)malloc(N * sizeof(float));
	hostB = (float*)malloc(N * sizeof(float));
	hostC = (float*)malloc(N * sizeof(float));

	float *deviceA, *deviceB, *deviceC;

	cudaMalloc((void**)&deviceA, N * sizeof(float));
	cudaMalloc((void**)&deviceB, N * sizeof(float));
	cudaMalloc((void**)&deviceC, N * sizeof(float));

	//Инициализация массивов случайными значениями 
	for (int i = 0; i < N; i++)
	{
		hostA[i] = rand();
		hostB[i] = rand();
	}

	cudaMemcpy(deviceA, hostA, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 


	myKernel << <blocksPerGrid, threadsPerBlock >> > (deviceA, deviceB, deviceC, N);

	cudaMemcpy(hostC, deviceC, N * sizeof(float), cudaMemcpyDeviceToHost);


	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop

	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop


	cout << "\t" << gpuTime; // Печатаем время

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);
}

void startKernel_CountRuntimeAndAllCopyTime(cudaDeviceProp devProp, dim3 threadsPerBlock, dim3 blocksPerGrid) {
	float *hostA, *hostB, *hostC;
	int N = devProp.maxThreadsPerBlock * BLOCKS_PER_GRID;

	hostA = (float*)malloc(N * sizeof(float));
	hostB = (float*)malloc(N * sizeof(float));
	hostC = (float*)malloc(N * sizeof(float));

	float *deviceA, *deviceB, *deviceC;

	cudaMalloc((void**)&deviceA, N * sizeof(float));
	cudaMalloc((void**)&deviceB, N * sizeof(float));
	cudaMalloc((void**)&deviceC, N * sizeof(float));

	//Инициализация массивов случайными значениями 
	for (int i = 0; i < N; i++)
	{
		hostA[i] = rand();
		hostB[i] = rand();
	}

	cudaEvent_t start, stop; // Описываем переменные типа cudaEvent_t
	float gpuTime = 0.0f;

	cudaEventCreate(&start); // Создаём событие начала выполнения ядра
	cudaEventCreate(&stop); // Создаём событие конца выполнения ядра 
	cudaEventRecord(start, 0); //Привязываем событие start к текущему месту 

	cudaMemcpy(deviceA, hostA, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, N * sizeof(float), cudaMemcpyHostToDevice);

	myKernel << <blocksPerGrid, threadsPerBlock >> > (deviceA, deviceB, deviceC, N);

	cudaMemcpy(hostC, deviceC, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту

	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop

	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop


	cudaEventDestroy(start); // Уничтожаем событие start 
	cudaEventDestroy(stop);	// Уничтожаем событие stop

	cout << "\t" << gpuTime; // Печатаем время

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);
}

// тело основной программы
int main(void) {
	setlocale(LC_ALL, "Russian");

	//переменная для хранения параметров GPU-устройства 
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	cout << "Нитей в блоке"
		<< "\tБлоков в сетке"
		<< "\tI"
		<< "\tII"
		<< "\tIII"
		<< endl;

	dim3 blocksPerGrid = BLOCKS_PER_GRID;
	dim3 threadsPerBlock = devProp.maxThreadsPerBlock;

	//Собрать данные о времени работы методов
	for (int i = 2; i <= 128; i *= 2) {

		cout << threadsPerBlock.x << "\t\t" << blocksPerGrid.x << "\t";

		startKernel_CountRuntime(devProp, threadsPerBlock, blocksPerGrid);

		startKernel_CountRuntimeAndDeviceToHostCopyTime(devProp, threadsPerBlock, blocksPerGrid);

		startKernel_CountRuntimeAndAllCopyTime(devProp, threadsPerBlock, blocksPerGrid);

		cout << endl;

		blocksPerGrid = BLOCKS_PER_GRID * i;
		threadsPerBlock = devProp.maxThreadsPerBlock / i;
	}

	startCPUCode_CountRuntime(devProp);

	cout << endl << "I - Время исполнения ядра, мс" << endl
		<< "II - Время исполнения ядра + копирование DeviceToHost, мс" << endl
		<< "III - Копирование HostToDevice + время исполнения ядра + копирование DeviceToHost, мс" << endl;

	//ждем нажатие клавиши 
	system("pause");
	return 0;
}