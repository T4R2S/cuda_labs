#include <iostream>
#include <windows.h>

using namespace std;

int main() {
	setlocale(LC_ALL, "Russian");

	cout << "test message" << endl;

	HINSTANCE h; //Дескриптор загружаемой dll (“указатель на приложение”)

	if ((h = LoadLibrary("lab_4")) == NULL) {
		cout << "Произошел казус" << endl;
	}
	else {
		//Объявление указателя на функцию, вызываемую из DLL
		void(*DllFunc) (float *a, float *b, int *N, float *test, float *time);
		DllFunc = (void(*) (float *a, float *b, int *N, float *test, float *time))
			GetProcAddress(h, "GpuIntegralCalc"); //получаем адрес функции SumGPU


		float a = 1;
		float b = 10;
		int N = 4096;
		float test = 0;
		float time = 0;

		DllFunc(&a, &b, &N, &test, &time);

		cout << time << endl;
		cout << test << endl;

		FreeLibrary(h);
	}

	system("pause");
	return 0;
}