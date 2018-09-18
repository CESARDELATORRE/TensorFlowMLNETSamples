using Microsoft.ML.Runtime.Api;

namespace ImageClassification.ImageData
{
    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }
}
