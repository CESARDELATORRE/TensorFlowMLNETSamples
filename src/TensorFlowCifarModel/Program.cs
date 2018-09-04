using System;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.IO;

namespace TensorFlowCifarModelScoring
{
    // Currently, need to install NuGet packages from MyGet: 
    // https://dotnet.myget.org/F/dotnet-core/api/v3/index.json 
    class Program
    {
        static FileInfo currentAssemblyLocation = new FileInfo(typeof(Program).Assembly.Location);
        static private readonly string _rootDir = currentAssemblyLocation.Directory.FullName;
        static private readonly string _dataRoot = Path.Combine(_rootDir, "data");

        static void Main(string[] args)
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model.pb";
            var dataFile = GetDataPath("tags/images.tsv");
            var tagsFolder = Path.GetDirectoryName(dataFile);
            var imagesFolder = Path.Combine(_dataRoot, "images");

            var pipeline = new LearningPipeline();
            pipeline.Add(new Microsoft.ML.Data.TextLoader(dataFile).CreateFrom<CifarData>(useHeader: false));
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

            pipeline.Add(new ImagePixelExtractor(("ImageCropped", "Input"))
            {
                UseAlpha = false,
                InterleaveArgb = true
            });

            pipeline.Add(new TensorFlowScorer()
            {
                ModelFile = model_location,
                InputColumns = new[] { "Input" },
                OutputColumn = "Output"
            });

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", inputColumns: "Output"));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Hack/workaround for a bug in ML.NET 0.6 preview. 
            // These two lines shouldn't be needed after the bug is fixed
            // These two lines are not needed if referencing the ML.NET OSS code projects directly..
            var hackArguments = new TensorFlowTransform.Arguments();
            var hackImageLoaderTransform= new ImageLoaderTransform.Arguments();

            var model = pipeline.Train<CifarData, CifarPrediction>();
            string[] scoreLabels;
            model.TryGetScoreLabelNames(out scoreLabels);

            //Test Scoring

            CifarPrediction prediction = model.Predict(new CifarData()
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

    public class CifarData
    {
        [Column("0")]
        public string ImagePath;

        [Column("1")]
        public string Label;
    }

    public class CifarPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }
}
