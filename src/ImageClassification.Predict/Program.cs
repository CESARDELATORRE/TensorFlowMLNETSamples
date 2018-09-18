using System;
using System.IO;
using System.Threading.Tasks;
using ImageClassification.Model;

namespace ImageClassification.Predict
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Running inside Visual Studio, $SolutionDir/assets is automatically passed as argument
            // If you execute from the console, pass as argument the location of the assets folder
            // Otherwise, it will search for assets in the executable's folder
            var assetsPath = args.Length > 0 ? args[0] : ModelHelpers.GetAssetsPath();

            var tagsTsv = Path.Combine(assetsPath, "inputs", "data", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "data");
            var imageClassifierZip = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");

            try
            {
                var modelEvaluator = new ModelEvaluator(tagsTsv, imagesFolder, imageClassifierZip);
                await modelEvaluator.Evaluate();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception: {ex.Message}");
            }
            Console.ReadKey();
        }
    }
}
