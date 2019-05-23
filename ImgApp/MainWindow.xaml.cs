using System;
using System.Linq;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Threading.Tasks;

namespace ImgApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Im_Click(object sender, RoutedEventArgs e)
        {
            ImButton.IsEnabled = false;
            CpuGauss();
            ImButton.IsEnabled = true;
        }

        public void CpuGauss()
        {
            var width = (int)Img.Width;
            var height = (int)Img.Height;

            var pixels = new double[width * height];

            var stopwatch = Stopwatch.StartNew();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var spot1 = Math.Exp(-(Math.Pow((y - 44.0) / height, 2) + Math.Pow((x - 42.0) / width, 2)) / 0.0055);
                    var spot2 = Math.Exp(-(Math.Pow((y - 244.0) / height, 2) + Math.Pow((x - 242.0) / width, 2)) / 0.0055);
                    var spot3 = Math.Exp(-(Math.Pow((y - 344.0) / height, 2) + Math.Pow((x - 342.0) / width, 2)) / 0.0055);

                    var val = 255 * (spot1 + spot2 + spot3);

                    pixels[y * height + x] = val;
                }
            }

            stopwatch.Stop();

            var max = pixels.Max();
            var min = pixels.Min();

            var bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);

            bitmap.Lock();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    IntPtr backbuffer = bitmap.BackBuffer;
                    backbuffer += y * bitmap.BackBufferStride;
                    backbuffer += x * sizeof(Int32);

                    var val = pixels[y * height + x];

                    //нормализация (проецирование данных в диапозон от 0 до 255)
                    val = ((val - min) / (max - min)) * 255;

                    var color = System.Drawing.Color.FromArgb((int)val, 0, 0).ToArgb();
                    Marshal.WriteInt32(backbuffer, color);
                }
            }

            bitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
            bitmap.Unlock();

            Img.Source = bitmap;
            Console.Text = "Im_Click Complete " + Environment.NewLine + "Время: " + stopwatch.ElapsedMilliseconds + "мс";
        }

        private void Im_dll_Click(object sender, RoutedEventArgs e)
        {
            ImDllButton.IsEnabled = false;
            GpuGauss();
            ImDllButton.IsEnabled = true;
        }

        public void GpuGauss()
        {
            var width = (int)Img.Width;
            var height = (int)Img.Height;

            var pixels = new float[width * height];
            var time = 0.0f;
            CudaLib.GpuGaussCalc(pixels, ref time);

            var max = pixels.Max();
            var min = pixels.Min();

            var bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);

            bitmap.Lock();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    IntPtr backbuffer = bitmap.BackBuffer;
                    backbuffer += y * bitmap.BackBufferStride;
                    backbuffer += x * sizeof(Int32);

                    var val = pixels[y * height + x];

                    //нормализация (проецирование данных в диапозон от 0 до 255)
                    val = (val - min) / (max - min) * 255;

                    var color = System.Drawing.Color.FromArgb((int)val, 0, 0).ToArgb();
                    Marshal.WriteInt32(backbuffer, color);
                }
            }

            bitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
            bitmap.Unlock();

            Img.Source = bitmap;
            Console.Text = "Im_dll_Click Complete " + Environment.NewLine + "Время: " + time + "мс";
        }

        private void Tom_cu_Click(object sender, RoutedEventArgs e)
        {
            TomCuButton.IsEnabled = false;
            Img.Source = GpuTom((int)Img.Width, (int)Img.Height);
            Console.Text = "Tom_cu_Click";
            TomCuButton.IsEnabled = true;
        }

        private BitmapSource GpuTom(int width, int height)
        {
            //Чтение файла
            var tomographResult = FileReader.Read(@"C:\Users\Роман\Downloads\Проекции для задания 6\subgroup_1\projections_variant_3_subgroup_1\model_v3_512_100_1_8_4.txt");

            var stopwatch = Stopwatch.StartNew();

            //Расчет проекций
            var pixels = CudaLib.BackProjection(tomographResult);

            stopwatch.Stop();

            Dispatcher.Invoke(() => Console.Text = "Время: " + stopwatch.ElapsedMilliseconds + "мс" + Environment.NewLine);

            //Подготовка битовой карты для отображения изображения
            var bitmap = GetBitmap(pixels, width, height);

            return bitmap;
        }

        private async void Tom_Click(object sender, RoutedEventArgs e)
        {
            TomButton.IsEnabled = false;
            var width = (int)Img.Width;
            var height = (int)Img.Height;

            Img.Source = await CpuTom(width, height);
            Console.Text += "Tom_Click Complete";
            TomButton.IsEnabled = true;
        }

        private float[,] BackProjection(TomographResult tomographResult)
        {
            var NN = tomographResult.SampleCount;
            var pixels = new float[NN, NN];

            foreach (var projection in tomographResult.Projections)
            {
                var angle = projection.Key;

                var sinGL = Math.Sin(Math.PI / 180 * angle);
                var cosGL = Math.Cos(Math.PI / 180 * angle);


                for (int i = 0; i < NN; i++)
                {
                    for (int j = 0; j < NN; j++)
                    {
                        var v = (i - (NN / 2)) * cosGL + (j - (NN / 2)) * sinGL;
                        var r = Math.Round(v) + NN / 2;

                        if ((r < (NN - 1)) && (r > -1))
                        {
                            pixels[i, j] = pixels[i, j] + projection.Value[(int)r];
                        }
                    }
                }
            }

            return pixels;
        }

        private async Task<BitmapSource> CpuTom(int width, int height)
        {
            return await Task.Run(() =>
            {
                //Чтение файла
                var tomographResult = FileReader.Read(@"C:\Users\Роман\Downloads\Проекции для задания 6\subgroup_1\projections_variant_3_subgroup_1\model_v3_512_100_1_8_4.txt");

                var stopwatch = Stopwatch.StartNew();

                //Расчет проекций
                var pixels = BackProjection(tomographResult);

                stopwatch.Stop();

                Dispatcher.Invoke(() => Console.Text = "Время: " + stopwatch.ElapsedMilliseconds + "мс" + Environment.NewLine);

                //Подготовка битовой карты для отображения изображения
                var bitmap = GetBitmap(pixels, width, height);

                return bitmap;
            });
        }

        private BitmapSource GetBitmap(float[,] pixels, int width, int height)
        {
            var bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);

            bitmap.Lock();

            var max = pixels.Cast<float>().Max();
            var min = pixels.Cast<float>().Min();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    IntPtr backbuffer = bitmap.BackBuffer;
                    backbuffer += y * bitmap.BackBufferStride;
                    backbuffer += x * sizeof(int);

                    var val = Math.Round(pixels[y, x]);

                    //нормализация (проецирование данных в диапозон от 0 до 255)
                    val = (val - min) / (max - min) * 255;

                    var color = System.Drawing.Color.FromArgb((int)val, 0, 0).ToArgb();
                    Marshal.WriteInt32(backbuffer, color);
                }
            }

            bitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
            bitmap.Unlock();
            bitmap.Freeze();

            return bitmap;
        }
    }
}
