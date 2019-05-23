#pragma once

extern "C" {
	__declspec(dllexport) void GpuGaussCalc(float *hostIm, float time);
}