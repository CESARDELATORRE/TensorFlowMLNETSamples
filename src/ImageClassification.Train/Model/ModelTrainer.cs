using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Transforms.TensorFlow;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ImageClassification.ImageData;
using static ImageClassification.Model.ModelHelpers;

namespace ImageClassification.Model
{
    public class ModelTrainer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string inputModelLocation;
        private readonly string outputModelLocation;

        public ModelTrainer(string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.inputModelLocation = inputModelLocation;
            this.outputModelLocation = outputModelLocation;
        }

        private struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const float scale = 1;
            public const bool channelsLast = true;
        }

        private struct InceptionSettings
        {
            // for checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string inputTensorName = "input";

            // output tensor name: in this case, is the node before the last one (not softmax, but softmax2_pre_activation).
            public const string outputTensorName = "softmax2_pre_activation";
        }

        public async Task BuildAndTrain()
        {
            var learningPipeline = BuildModel(dataLocation, imagesFolder, inputModelLocation);

            var model = TrainModel(learningPipeline);
            //var predictions = TestModel(dataLocation, imagesFolder, model).ToArray();
            //ShowPredictions(predictions);

            await SaveModel(model, outputModelLocation);
        }

        private async Task SaveModel(PredictionModel<ImageNetData, ImageNetPrediction> model, string modelLocation)
        {
            if (!string.IsNullOrEmpty(modelLocation))
            {
                ConsoleWriteHeader("Save model to local file");
                ModelHelpers.DeleteAssets(modelLocation);
                await model.WriteAsync(modelLocation);
                Console.WriteLine($"Model saved: {modelLocation}");
            }
        }

        protected PredictionModel<ImageNetData, ImageNetPrediction> TrainModel(LearningPipeline pipeline)
        {
            ConsoleWriteHeader("Training classification model");

            // Initialize TensorFlow engine
            TensorFlowUtils.Initialize();

            var model = pipeline.Train<ImageNetData, ImageNetPrediction>();
            return model;
        }

        protected LearningPipeline BuildModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            const bool convertPixelsToFloat = true;
            const bool ignoreAlphaChannel = false;

            ConsoleWriteHeader("Build model pipeline");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Inception model location: {modelLocation}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}, image scale: {ImageNetSettings.scale}");

            var pipeline = new LearningPipeline();

            // TextLoader loads tsv file, containing image file location and label 
            pipeline.Add(new Microsoft.ML.Legacy.Data.TextLoader(dataLocation).CreateFrom<ImageNetData>(useHeader: false));

            // ImageLoader reads input images
            pipeline.Add(new ImageLoader((nameof(ImageNetData.ImagePath), "ImageReal"))
            {
                ImageFolder = imagesFolder
            });

            // ImageResizer is used to resize input image files
            // to the size used by the Neural Network
            pipeline.Add(new ImageResizer(("ImageReal", "ImageCropped"))
            {
                ImageHeight = ImageNetSettings.imageHeight,
                ImageWidth = ImageNetSettings.imageWidth,
                Resizing = ImageResizerTransformResizingKind.IsoCrop
            });

            // ImagePixelExtractor is used to process the input image files
            // according to the requirements of the Deep Neural Network
            // This step is the perfect place to make specific image transformations,
            // like normalizing pixel values (Pixel * scale / offset). 
            // This kind of image pre-processing is common when dealing with images used in DNN
            pipeline.Add(new ImagePixelExtractor(("ImageCropped", InceptionSettings.inputTensorName))
            {
                UseAlpha = ignoreAlphaChannel, // channel = (red, green, blue)
                InterleaveArgb = ImageNetSettings.channelsLast, // (width x height x channel)
                Convert = convertPixelsToFloat,
                Offset = ImageNetSettings.mean, // pixel normalization
                Scale = ImageNetSettings.scale // pixel normalization
            });

            // TensorFlowScorer is used to get the activation map before the last output of the Neural Network
            // This activation map is used as a image vector featurizer 
            pipeline.Add(new TensorFlowScorer()
            {
                ModelFile = modelLocation,
                InputColumns = new[] { InceptionSettings.inputTensorName },
                OutputColumns = new[] { InceptionSettings.outputTensorName }
            });


            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", inputColumns: InceptionSettings.outputTensorName));
            pipeline.Add(new TextToKeyConverter(nameof(ImageNetData.Label)));

            // At this point, there are two inputs for the learner: 
            // * Features: input image vector feaures
            // * Label: label fom the input file
            // In this case, we use SDCA for classifying images using Label / Features columns. 
            // Other multi-class classifier may be used instead of SDCA
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            return pipeline;
        }
    }
}
