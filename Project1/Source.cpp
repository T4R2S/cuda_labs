#include <iostream>
#include <windows.h>

using namespace std;

int main() {
	setlocale(LC_ALL, "Russian");

	HINSTANCE h; //Дескриптор загружаемой dll (“указатель на приложение”)

	if ((h = LoadLibrary("lab_4")) == NULL) {
		cout << "Произошел казус" << endl;
	}
	else {
		//Объявление указателя на функцию, вызываемую из DLL
		void(*DllFunc) (float a, float b, int N, float *result, float *time);
		DllFunc = (void(*) (float a, float b, int N, float *result, float *time))
			GetProcAddress(h, "GpuIntegralCalc"); //получаем адрес функции SumGPU


		float a = 1;
		float b = 10;
		int N = 4096*128;
		float result = 0;
		float time = 0;

		DllFunc(a, b, N, &result, &time);

		cout << "Время работы, мс: " << time << endl;
		cout << "Ответ: " << result << endl;

		FreeLibrary(h);
	}

	system("pause");
	return 0;
}