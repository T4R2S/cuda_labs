#pragma once

extern "C" {
	__declspec(dllexport) void GpuGaussCalc(float *hostIm, float *time);
}

extern "C" {
	__declspec(dllexport) void BackProjection(float* host_discretP, int host_NN, int host_M, float host_dQ, float* host_XY, float *time);
}