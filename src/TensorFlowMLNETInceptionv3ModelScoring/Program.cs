using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;
using System;
using System.IO;

namespace TensorFlowMLNETInceptionv3ModelScoring
{
    class Program
    {
        static FileInfo currentAssemblyLocation = new FileInfo(typeof(Program).Assembly.Location);
        static private readonly string _rootDir = currentAssemblyLocation.Directory.FullName;
        static private readonly string _dataRoot = Path.Combine(_rootDir, "data");
        const float mean = 117;
        const float scale = 1;
        const int imageHeight = 224;
        const int imageWidth = 224;
        const bool convertPixelsToFloat = true;
        const bool ignoreAlphaChannel = false;
        const bool channelsLast = true;
        const string inputTensorName = "input";
        const string outputTensorName = "softmax2_pre_activation";

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var model_location = "model/tensorflow_inception_graph.pb";
            var dataFile = GetDataPath("model/imagenet.tsv");
            var tagsFolder = Path.GetDirectoryName(dataFile);
            var imagesFolder = Path.Combine(_dataRoot, "images");

            var pipeline = new LearningPipeline();
            pipeline.Add(new Microsoft.ML.Data.TextLoader(dataFile).CreateFrom<ImageNetData>(useHeader: false));
            pipeline.Add(new ImageLoader(("ImagePath", "ImageReal"))
            {
                ImageFolder = imagesFolder
            });

            pipeline.Add(new ImageResizer(("ImageReal", "ImageCropped"))
            {
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                Resizing = ImageResizerTransformResizingKind.IsoCrop
            });

            pipeline.Add(new ImagePixelExtractor(("ImageCropped", inputTensorName))
            {
                UseAlpha = ignoreAlphaChannel,
                InterleaveArgb = channelsLast,
                Convert = convertPixelsToFloat,
                Offset = mean,
                Scale = scale
            });

            pipeline.Add(new TensorFlowScorer()
            {
                ModelFile = model_location,
                InputColumns = new[] { inputTensorName },
                OutputColumn = outputTensorName
            });

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", inputColumns: outputTensorName));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Hack/workaround for a bug in ML.NET preview. 
            // These two lines shouldn't be needed after the bug is fixed
            // These two lines are not needed if referencing the ML.NET OSS code projects directly..
            //var hackArguments = new TensorFlowTransform.Arguments();
            //var hackImageLoaderTransform = new ImageLoaderTransform.Arguments();

            TensorFlowUtils.Initialize();

            var model = pipeline.Train<ImageNetData, ImageNetPrediction>();
            string[] scoreLabels;
            model.TryGetScoreLabelNames(out scoreLabels);

            //Test Scoring

            ImageNetPrediction prediction = model.Predict(new ImageNetData()
            {
                ImagePath = GetDataPath("images/banana.jpg")
            });

            Console.WriteLine("End of process");
        }

        static protected string GetDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(_dataRoot, name));
        }

    }

    public class ImageNetData
    {
        [Column("0")]
        public string ImagePath;

        [Column("1")]
        public string Label;
    }

    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }

}
