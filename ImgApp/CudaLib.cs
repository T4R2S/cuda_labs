using System.Runtime.InteropServices;

namespace ImgApp
{
    public class CudaLib
    {
        [DllImport("lab_5.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void GpuGaussCalc(float[] hostIm, ref float time);

        [DllImport("lab_5.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void BackProjection(float[] host_discretP, int host_NN, int host_M, float host_dQ, float[] pixels, ref float time);
    }
}
