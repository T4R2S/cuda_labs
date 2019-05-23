using NUnit.Framework;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ImgAppTests.UnitTests
{
    [TestFixture]
    public class FileReaderTests
    {
        [Test]
        public void Read()
        {
            var fileLines = File.ReadLines(@"C:\Users\Роман\Downloads\Проекции для задания 6\subgroup_1\projections_variant_3_subgroup_1\model_v3_512_100_1_8_4.txt");

            if (fileLines.Count() == 0)
                throw new System.Exception("пустой файл");

            if (fileLines.First() != "data")
                throw new System.Exception("неверный формат данных");

            var operationCount = int.Parse(fileLines.ElementAt(1));
            var projectionCount = int.Parse(fileLines.ElementAt(2));
            var angleStep = float.Parse(fileLines.ElementAt(3), CultureInfo.InvariantCulture);

            var projectionsWithAngles = fileLines.Skip(5);

            var result = new Dictionary<float, float[]>();
            var angle = 0f;

            for (int i = 0; i < projectionsWithAngles.Count(); i += 2)
            {
                var projection = projectionsWithAngles
                    .ElementAt(i)
                    .Split(' ')
                    .Where(x => string.IsNullOrWhiteSpace(x) == false)
                    .Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();

                result.Add(angle, projection);

                angle += angleStep;
            }

            Assert.AreEqual(projectionCount, result.Count());
            Assert.AreEqual(operationCount, result[1.8f].Count());
            Assert.AreEqual(4.73673186483237E-0004f, result[1.8f][0]);
        }

    }
}
