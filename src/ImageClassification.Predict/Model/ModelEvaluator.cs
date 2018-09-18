using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Transforms.TensorFlow;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ImageClassification.ImageData;
using static ImageClassification.Model.ModelHelpers;
using System.IO;

namespace ImageClassification.Model
{
    public class ModelEvaluator
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;

        public ModelEvaluator(string dataLocation, string imagesFolder, string modelLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
        }

        public async Task Evaluate()
        {
            PredictionModel<ImageNetData, ImageNetPrediction> model = await LoadModel();

            PredictDataUsingModel(dataLocation, imagesFolder, model).ToArray();

            EvaluateModel(dataLocation, model);
        }

        private async Task<PredictionModel<ImageNetData, ImageNetPrediction>> LoadModel()
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {modelLocation}");

            // Initialize TensorFlow engine
            TensorFlowUtils.Initialize();

            return await PredictionModel.ReadAsync<ImageNetData, ImageNetPrediction>(modelLocation);
        }

        protected IEnumerable<ImageNetData> PredictDataUsingModel(string testLocation, string imagesFolder, PredictionModel<ImageNetData, ImageNetPrediction> model)
        {
            ConsoleWriteHeader("Classificate images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine(" ");

            model.TryGetScoreLabelNames(out string[] labels);
            var testData = ImageNetData.ReadFromCsv(testLocation, imagesFolder).ToList();
            // add an extra image to "really" test the model
            testData = testData.Concat(new[] { new ImageNetData() { ImagePath = Path.Combine(imagesFolder, "teddy5.jpg") } }).ToList();

            foreach (var sample in testData)
            {
                var probs = model.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                };
                (imageData.Label, imageData.Probability) = GetLabel(labels, probs);
                imageData.ConsoleWriteLine();
                yield return imageData;
            }
        }

        protected void EvaluateModel(string testLocation, PredictionModel<ImageNetData, ImageNetPrediction> model)
        {
            ConsoleWriteHeader("Metrics for Image Classification");
            var evaluator = new ClassificationEvaluator();
            var testDataSource = new TextLoader(testLocation).CreateFrom<ImageData.ImageNetData>();
            ClassificationMetrics metrics = evaluator.Evaluate(model, testDataSource);

            // Log loss is nearly zero. The lower the better, so in this case can't be better
            // This is an "ideal" situation, usually you get higher values
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");
        }
    }
}
