using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace ImgApp
{
    public class CudaLib
    {
        [DllImport("lab_5.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void GpuGaussCalc(float[] hostIm, ref float time);

        [DllImport("lab_5.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern float[,] BackProjection(TomographResult tomographResult);
    }
}
