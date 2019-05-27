using System;
using System.Linq;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Threading.Tasks;
using Microsoft.Win32;

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

        private async void Im_Click(object sender, RoutedEventArgs e)
        {
            ImButton.IsEnabled = false;
            Img.Source = await CpuGauss((int)Img.Width, (int)Img.Height);
            Console.Text += "Im_Click Complete";
            ImButton.IsEnabled = true;
        }

        private async Task<BitmapSource> CpuGauss(int width, int height)
        {
            return await Task.Run(() =>
            {
                var pixels = new float[width, height];

                var stopwatch = Stopwatch.StartNew();

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var spot1 = Math.Exp(-(Math.Pow((y - 44.0) / height, 2) + Math.Pow((x - 42.0) / width, 2)) / 0.0055);
                        var spot2 = Math.Exp(-(Math.Pow((y - 244.0) / height, 2) + Math.Pow((x - 242.0) / width, 2)) / 0.0055);
                        var spot3 = Math.Exp(-(Math.Pow((y - 344.0) / height, 2) + Math.Pow((x - 342.0) / width, 2)) / 0.0055);

                        var val = 255 * (spot1 + spot2 + spot3);

                        pixels[y, x] = (float)val;
                    }
                }

                stopwatch.Stop();

                Dispatcher.Invoke(() => Console.Text = "Время: " + stopwatch.ElapsedMilliseconds + "мс" + Environment.NewLine);

                return GetBitmap(pixels, width, height);
            });
        }

        private async void Im_dll_Click(object sender, RoutedEventArgs e)
        {
            ImDllButton.IsEnabled = false;
            Img.Source = await GpuGauss((int)Img.Width, (int)Img.Height);
            Console.Text += "Im_dll_Click Complete ";
            ImDllButton.IsEnabled = true;
        }

        private async Task<BitmapSource> GpuGauss(int width, int height)
        {
            return await Task.Run(() =>
            {
                var result = new float[width * height];
                var time = 0.0f;
                CudaLib.GpuGaussCalc(result, ref time);

                var pixels = new float[height, width];

                for (int i = 0; i < height; i++)
                    for (int j = 0; j < width; j++)
                        pixels[i, j] = result[j + i * height];

                Dispatcher.Invoke(() => Console.Text = "Время: " + time + "мс" + Environment.NewLine);
                return GetBitmap(pixels, width, height, true);
            });
        }

        private async void Tom_cu_Click(object sender, RoutedEventArgs e)
        {
            TomCuButton.IsEnabled = false;
            Img.Source = await GpuTom((int)Img.Width, (int)Img.Height);
            Console.Text += "Tom_cu_Click Complete";
            TomCuButton.IsEnabled = true;
        }

        private async Task<BitmapSource> GpuTom(int width, int height)
        {
            return await Task.Run(async () =>
            {
                //Чтение файла
                var fileDialog = new OpenFileDialog();
                var filePath = string.Empty;

                if (fileDialog.ShowDialog() == true)
                    filePath = fileDialog.FileName;
                else
                    return null;

                var tomographResult = await FileReader.Read(filePath);

                var time = 0.0f;

                //Расчет проекций
                float[] projetions = tomographResult.Projections.SelectMany(x => x.Value).ToArray();
                float[] result = new float[height * width];

                CudaLib.BackProjection(projetions, tomographResult.SampleCount, tomographResult.ProjectionCount, tomographResult.AngleStep, result, ref time);

                //перевод результата в двумерный массив
                var pixels = new float[height, width];

                for (int i = 0; i < height; i++)
                    for (int j = 0; j < width; j++)
                        pixels[i, j] = result[j + i * height];

                Dispatcher.Invoke(() => Console.Text = "Время: " + time + "мс" + Environment.NewLine);

                //Подготовка битовой карты для отображения изображения
                var bitmap = await GetBitmap(pixels, width, height, true);

                return bitmap;
            });
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

                var sinGL = Math.Sin(Math.PI / 180.0f * angle);
                var cosGL = Math.Cos(Math.PI / 180.0f * angle);


                for (int i = 0; i < NN; i++)
                {
                    for (int j = 0; j < NN; j++)
                    {
                        var v = (j - (NN / 2.0f)) * cosGL + (i - (NN / 2.0f)) * sinGL;
                        var r = Math.Round(v) + NN / 2.0f;

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
            return await Task.Run(async () =>
            {
                //Чтение файла
                var fileDialog = new OpenFileDialog();
                var filePath = string.Empty;

                if (fileDialog.ShowDialog() == true)
                    filePath = fileDialog.FileName;
                else
                    return null;

                var tomographResult = await FileReader.Read(filePath);

                var stopwatch = Stopwatch.StartNew();

                //Расчет проекций
                var pixels = BackProjection(tomographResult);

                stopwatch.Stop();

                Dispatcher.Invoke(() => Console.Text = "Время: " + stopwatch.ElapsedMilliseconds + "мс" + Environment.NewLine);

                //Подготовка битовой карты для отображения изображения
                var bitmap = await GetBitmap(pixels, width, height);

                return bitmap;
            });
        }

        private async Task<BitmapSource> GetBitmap(float[,] pixels, int width, int height, bool cudaColor = false)
        {
            return await Task.Run(() =>
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

                        var red = cudaColor ? 0 : (int)val;
                        var green = (int)val;
                        var blue = cudaColor ? 0 : (int)val;

                        var color = System.Drawing.Color.FromArgb(red, green, blue).ToArgb();
                        Marshal.WriteInt32(backbuffer, color);
                    }
                }

                bitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
                bitmap.Unlock();
                bitmap.Freeze();

                return bitmap;
            });
        }
    }
}
