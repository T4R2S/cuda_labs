using System.Collections.Generic;

namespace ImgApp
{
    public class TomographResult
    {
        public TomographResult()
        {
            Projections = new Dictionary<float, float[]>();
        }

        public int SampleCount { get; set; }
        public int ProjectionCount { get; set; }
        public float AngleStep { get; set; }
        public Dictionary<float, float[]> Projections { get; set; }
    }
}
