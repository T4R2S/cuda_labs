
#include <iostream> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <math.h>
using namespace std;//использовано пространство имен std


__global__ void myKernel(float *A, float *B, float *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		C[i] = A[i] + B[i];
}

// тело основной программы
int main(void) {
	setlocale(LC_ALL, "Russian");

	//переменная для хранения параметров GPU-устройства 
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	float *hostA, *hostB, *hostC;
	int N = devProp.maxThreadsPerBlock * 512; //512 - блоков в сетке

	cout << devProp.maxThreadsPerBlock << endl;

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

	myKernel << <512*4, devProp.maxThreadsPerBlock/4 >> > (deviceA, deviceB, deviceC, N);

	cudaEventRecord(stop, 0); //Привязываем событие stop к текущему месту
	
	cudaEventSynchronize(stop); //Ждем реального окончания выполнения ядра, используя 
								//возможность синхронизации по событию stop
	
	cudaEventElapsedTime(&gpuTime, start, stop); // Запрос времени между событиями start 
												 // и stop
	
	cout << " Время работы мс: " << gpuTime << endl; // Печатаем время

	cudaEventDestroy(start); // Уничтожаем событие start cudaEventDestroy(stop); 
							 // Уничтожаем событие stop

	cudaMemcpy(hostC, deviceC, N * sizeof(float), cudaMemcpyDeviceToHost);

	// Вывод на экран результата вычислений
	for (int i = 0; i < 10; i++) {
		cout << hostA[i] << "+" << hostB[i] << "=" << hostC[i] << endl;
	}

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);

	//ждем нажатие клавиши 
	system("pause");
	return 0;
}