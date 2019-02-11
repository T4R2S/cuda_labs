// ������������ ����� 
#include <iostream> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
using namespace std;//������������ ������������ ���� std

// ���� �������� ���������
int main(void) {
	//�������, ����������� ��������� ���������� ������������� ���������
	setlocale(LC_ALL, "Russian");

	//���������� ��� ���������� ��������� 
	int deviceCount;

	//���������� ��� �������� ���������� GPU-���������� 
	cudaDeviceProp devProp;

	//��������� ���������� � ���������� ��������� 
	cudaGetDeviceCount(&deviceCount);
	cout << "������� ���������: " << deviceCount << endl;

	//������� ��������� ������� ���������� 
	for (int device = 0; device < deviceCount; device++)
	{
		//��������� ���������� ��������� 
		cudaGetDeviceProperties(&devProp, device);
		//����� ���������� �� ����� ����� ����� cout 
		cout << "���������� � " << device << endl;
		cout << "��� ���������� " << devProp.name << endl;

		cout << "����� ���������� ������ GPU ����������: " << devProp.totalGlobalMem << endl;
		cout << "����� ����������� ������ � ����� �����: " << devProp.sharedMemPerBlock << endl;
		cout << "���������� 32-��������� ��������� � ����� �����: " << devProp.regsPerBlock << endl;
		cout << "���������� ����� � �����: " << devProp.warpSize << endl;
		cout << "������������ ��� (pitch) � ������, ����������� ��������� ������������� ������: " << devProp.memPitch << endl;
		cout << "������������ ���������� ����� � ����� �����: " << devProp.maxThreadsPerBlock << endl;

		for (int i = 1; i <= 3; i++)
			cout << "������������ ���������� ����� ����� " << i << "-�� �����: " << devProp.maxThreadsDim[i] << endl;

		for (int i = 1; i <= 3; i++)
			cout << "������������ ���������� ������ � ����� ����� " << i << "-�� �����: " << devProp.maxGridSize[i] << endl;

		cout << "������������ �������������� ������ ���������� �������: " << devProp.maxTexture1D << endl;

		for (int i = 1; i <= 2; i++)
			cout << "������������ �������������� ������ " << i << "-�� ����� ������� ���������� �������: " << devProp.maxTexture2D[i] << endl;

		for (int i = 1; i <= 3; i++)
			cout << "������������ �������������� ������ " << i << "-�� ����� ������� ���������� �������: " << devProp.maxTexture3D[i] << endl;

		cout << "������������ ������ ������� 1D �������� (dim): " << devProp.maxTexture1DLayered[0] << endl;
		cout << "������������ ������ ������� 1D �������� (layers): " << devProp.maxTexture1DLayered[1] << endl;

		cout << "������������ ������ ������� 2D �������� (dim1): " << devProp.maxTexture2DLayered[0] << endl;
		cout << "������������ ������ ������� 2D �������� (dim2): " << devProp.maxTexture2DLayered[1] << endl;
		cout << "������������ ������ ������� 2D �������� (layers): " << devProp.maxTexture2DLayered[2] << endl;

		cout << "����� ��������� ����������� ������: " << devProp.totalConstMem << endl;

		cout << "������� ����� ������ �������������� ������������ GPU ����������: " << devProp.major << endl;
		cout << "������� ����� ������ �������������� ������������ GPU ����������: " << devProp.minor << endl;

		cout << "�������� ������� CUDA � ���� � ����������: " << devProp.clockRate << endl;
		cout << "������������ ������ ��� �������: " << devProp.textureAlignment << endl;
		cout << "������������ ������ ��� ������������: " << devProp.surfaceAlignment << endl;
		cout << "����� �� ���������� ������������ ��������� ������� cudaMemcpy() � ����? " << (devProp.deviceOverlap ? "��" : "���") << endl;
		cout << "���������� �������� �����������, ����������� ����������� " << devProp.asyncEngineCount << endl;
		cout << "������������ �� ���������� ������������� ���������� ���������� ���� � ����� ���������? " << (devProp.concurrentKernels ? "��" : "���") << endl;
		cout << "���������� ����������������� � GPU: " << devProp.multiProcessorCount << endl;
		cout << "���������� �� ����������� �� ����� ���������� ���� �����������? " << (devProp.kernelExecTimeoutEnabled ? "��" : "���") << endl;
		cout << "���������� �������� ��������������� GPU? " << (devProp.integrated ? "��" : "���") << endl;
		cout << "����� �� ���������� ���������� ������ CPU �� �������� ������������ CUDA-����������? " << (devProp.canMapHostMemory ? "��" : "���") << endl;
		cout << "������ ���� ���������� ������: " << devProp.memoryBusWidth << endl;
		cout << "�������� ������� ������: " << devProp.memoryClockRate << endl;
		cout << "������ L2 ����: " << devProp.l2CacheSize << endl;
		cout << "����������� ������������� ���������������� ��������� ������������: " << devProp.unifiedAddressing << endl;
		cout << "ID PCI ���� ����������: " << devProp.pciBusID << endl;
		cout << "PCI ID ����������: " << devProp.pciDeviceID << endl;
		cout << "1, ���� ���������� ���������� Tesla � ������������ TCC �������: " << devProp.tccDriver << endl;
		cout << "���� �� ��������� ������ UVA: " << devProp.unifiedAddressing << endl;

	}

	//���� ������� ������� 
	system("pause");
	return 0;
}