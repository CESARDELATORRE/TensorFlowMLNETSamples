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
    // - TensorFlow https://www.tensorflow.org/ is a popular machine learning toolkit that enables training deep neural networks(and general numeric computations).
    // - This sample show the usage of the ML.NET "TensorFlowScorer" transform that enables taking an existing TensorFlow model, either trained by you 
    //   or downloaded from somewhere else, and get the scores from the TensorFlow model in ML.NET code.
    // - For now, these scores (numeric vectors) can only be used within a LearningPipeline as inputs to a learner.
    //   However, with the upcoming ML.NET APIs, the scores from the TensorFlow model will be directly accessible.
    // - The implementation of this mentioned "TensorFlowScorer" transform is based on code from TensorFlowSharp.
    //
    // Sample code: Specifically, this sample code when training with the pipeline, it generates a numeric vector for each image that you have in the folder "images" 
    // and correlates those numeric vectors with the types of objects/things provided in the "tags.tsv" file. 
    // After that, the model is trained with an SDCA classifier (StochasticDualCoordinateAscentClassifier) that uses that relationship between numeric vectors and labels/tags. 
    // so when using the model in a test or final app, you can classify any given image that is similar to any of the images used in the pipeline.
    // IMPORTANT: Note that the sample is only training with one image per type so the accuracy will be poor.
    // In order to get a good accuracy and better effectiveness when classifying images you'd need to train with a much larger volume of images per image-type. 

    class Program
    {
        static FileInfo currentAssemblyLocation = new FileInfo(typeof(Program).Assembly.Location);
        static private readonly string _rootDir = currentAssemblyLocation.Directory.FullName;
        static private readonly string _dataRoot = Path.Combine(_rootDir, ".");
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
            // TensorFlow model file's pathname
            var model_location = "model/tensorflow_inception_graph.pb";

            // File pathname to textfile with relationship between image-files and image-types (tags)
            var dataFile = GetDataPath("model/tags.tsv");

            var tagsFolder = Path.GetDirectoryName(dataFile);
            var imagesFolder = Path.Combine(_dataRoot, "images");

            var pipeline = new LearningPipeline();

            // Load the tags file into the pipeline
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
                OutputColumns = new[] { outputTensorName }
            });

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", inputColumns: outputTensorName));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            TensorFlowUtils.Initialize();

            var model = pipeline.Train<ImageNetData, ImageNetPrediction>();
            string[] scoreLabels;
            model.TryGetScoreLabelNames(out scoreLabels);

            //Test Scoring
            ImageNetPrediction prediction = model.Predict(new ImageNetData()
            {
                ImagePath = GetDataPath("images/violin.jpg")
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
