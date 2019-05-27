using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImgApp
{
    public class FileReader
    {

        public static async Task<TomographResult> Read(string path)
        {
            return await Task.Run(() =>
            {
                var fileLines = File.ReadLines(path);

                if (fileLines.Count() == 0)
                    throw new Exception("пустой файл");

                if (fileLines.First() != "data")
                    throw new Exception("неверный формат данных");

                var result = new TomographResult
                {
                    SampleCount = int.Parse(fileLines.ElementAt(1)),
                    ProjectionCount = int.Parse(fileLines.ElementAt(2)),
                    AngleStep = float.Parse(fileLines.ElementAt(3), CultureInfo.InvariantCulture)
                };

                var projectionsWithAngles = fileLines.Skip(5);

                var angle = 0f;

                for (int i = 0; i < projectionsWithAngles.Count(); i += 2)
                {
                    var projection = projectionsWithAngles
                        .ElementAt(i)
                        .Split(' ')
                        .Where(x => string.IsNullOrWhiteSpace(x) == false)
                        .Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();

                    result.Projections.Add(angle, projection);

                    angle += result.AngleStep;
                }

                return result;
            });
        }
    }
}
