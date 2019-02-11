// заголовочные файлы 
#include <iostream> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
using namespace std;//использовано пространство имен std

// тело основной программы
int main(void) {
	//функция, позволяющая правильно выводились кириллическую кодировку
	setlocale(LC_ALL, "Russian");

	//переменная под количество устройств 
	int deviceCount;

	//переменная для хранения параметров GPU-устройства 
	cudaDeviceProp devProp;

	//получение информации о количестве устройств 
	cudaGetDeviceCount(&deviceCount);
	cout << "Найдено устройств: " << deviceCount << endl;

	//выводим параметры каждого устройства 
	for (int device = 0; device < deviceCount; device++)
	{
		//получение параметров устройств 
		cudaGetDeviceProperties(&devProp, device);
		//вывод параметров на экран через поток cout 
		cout << "Устройство № " << device << endl;
		cout << "Имя устройства " << devProp.name << endl;

		cout << "Объём глобальной памяти GPU устройства: " << devProp.totalGlobalMem << endl;
		cout << "Объём разделяемой памяти в одном блоке: " << devProp.sharedMemPerBlock << endl;
		cout << "Количество 32-разрядных регистров в одном блоке: " << devProp.regsPerBlock << endl;
		cout << "Количество нитей в варпе: " << devProp.warpSize << endl;
		cout << "Максимальный шаг (pitch) в байтах, разрешенный функциями копированиями памяти: " << devProp.memPitch << endl;
		cout << "Максимальное количество нитей в одном блоке: " << devProp.maxThreadsPerBlock << endl;

		for (int i = 1; i <= 3; i++)
			cout << "Максимальное количество нитей вдоль " << i << "-го блока: " << devProp.maxThreadsDim[i] << endl;

		for (int i = 1; i <= 3; i++)
			cout << "Максимальное количество блоков в сетке вдоль " << i << "-го блока: " << devProp.maxGridSize[i] << endl;

		cout << "Максимальный поддерживаемый размер одномерных текстур: " << devProp.maxTexture1D << endl;

		for (int i = 1; i <= 2; i++)
			cout << "Максимальный поддерживаемый размер " << i << "-го блока массива двухмерных текстур: " << devProp.maxTexture2D[i] << endl;

		for (int i = 1; i <= 3; i++)
			cout << "Максимальный поддерживаемый размер " << i << "-го блока массива трехмерных текстур: " << devProp.maxTexture3D[i] << endl;

		cout << "Максимальный размер массива 1D текстуры (dim): " << devProp.maxTexture1DLayered[0] << endl;
		cout << "Максимальный размер массива 1D текстуры (layers): " << devProp.maxTexture1DLayered[1] << endl;

		cout << "Максимальный размер массива 2D текстуры (dim1): " << devProp.maxTexture2DLayered[0] << endl;
		cout << "Максимальный размер массива 2D текстуры (dim2): " << devProp.maxTexture2DLayered[1] << endl;
		cout << "Максимальный размер массива 2D текстуры (layers): " << devProp.maxTexture2DLayered[2] << endl;

		cout << "Объём имеющейся константной памяти: " << devProp.totalConstMem << endl;

		cout << "Старшая часть уровня вычислительных возможностей GPU устройства: " << devProp.major << endl;
		cout << "Младшая часть уровня вычислительных возможностей GPU устройства: " << devProp.minor << endl;

		cout << "Тактовая частота CUDA – ядер в килогерцах: " << devProp.clockRate << endl;
		cout << "Выравнивание памяти для текстур: " << devProp.textureAlignment << endl;
		cout << "Выравнивание памяти для поверхностей: " << devProp.surfaceAlignment << endl;
		cout << "может ли устройство одновременно выполнять функцию cudaMemcpy() и ядро? " << (devProp.deviceOverlap ? "да" : "нет") << endl;
		cout << "Количество операций копирования, выполняемых параллельно " << devProp.asyncEngineCount << endl;
		cout << "поддерживает ли устройство одновременное исполнение нескольких ядер в одном контексте? " << (devProp.concurrentKernels ? "да" : "нет") << endl;
		cout << "Количество мультипроцессоров в GPU: " << devProp.multiProcessorCount << endl;
		cout << "существует ли ограничение на время исполнения ядер устройством? " << (devProp.kernelExecTimeoutEnabled ? "да" : "нет") << endl;
		cout << "Устройство явялется интегрированным GPU? " << (devProp.integrated ? "да" : "нет") << endl;
		cout << "может ли устройство отображать память CPU на адресное пространство CUDA-устройства? " << (devProp.canMapHostMemory ? "да" : "нет") << endl;
		cout << "Ширина шины глобальной памяти: " << devProp.memoryBusWidth << endl;
		cout << "Тактовая частота памяти: " << devProp.memoryClockRate << endl;
		cout << "Размер L2 кэша: " << devProp.l2CacheSize << endl;
		cout << "Возможность использования унифицированного адресного пространства: " << devProp.unifiedAddressing << endl;
		cout << "ID PCI шины устройства: " << devProp.pciBusID << endl;
		cout << "PCI ID устройства: " << devProp.pciDeviceID << endl;
		cout << "1, если подключено устройство Tesla и используется TCC драйвер: " << devProp.tccDriver << endl;
		cout << "Есть ли поддержка режима UVA: " << devProp.unifiedAddressing << endl;

	}

	//ждем нажатие клавиши 
	system("pause");
	return 0;
}