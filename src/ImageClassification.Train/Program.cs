using System;
using System.IO;
using System.Threading.Tasks;
using ImageClassification.Model;

namespace ImageClassification.Train
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            // Running inside Visual Studio, $SolutionDir/assets is automatically passed as argument
            // If you execute from the console, pass as argument the location of the assets folder
            // Otherwise, it will search for assets in the executable's folder
            var assetsPath = args.Length > 0 ? args[0] : ModelHelpers.GetAssetsPath();

            var tagsTsv = Path.Combine(assetsPath, "inputs", "data", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "data");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            var imageClassifierZip = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");

            try
            {
                var modelBuilder = new ModelTrainer(tagsTsv, imagesFolder, inceptionPb, imageClassifierZip);
                await modelBuilder.BuildAndTrain();

                //var modelEvaluator = new ModelEvaluator(
                //    ModelHelpers.GetAssetsPath("data", "tags.tsv"),
                //    ModelHelpers.GetAssetsPath("images"),
                //    ModelHelpers.GetAssetsPath("model", "imageClassifier.zip"));
                //await modelEvaluator.Evaluate();
            } catch (Exception ex)
            {
                Console.WriteLine($"Exception: {ex.Message}");
            }
            Console.ReadKey();
        }
    }
}
