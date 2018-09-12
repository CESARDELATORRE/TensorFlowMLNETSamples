using System;
using System.Threading.Tasks;
using TensorFlowMLNETInceptionv3ModelScoring.Model;

namespace TensorFlowMLNETInceptionv3ModelScoring
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                var modelBuilder = new ModelTrainer(
                   ModelHelpers.GetAssetsPath("data", "tags.tsv"),
                   ModelHelpers.GetAssetsPath("images"),
                   ModelHelpers.GetAssetsPath("model", "tensorflow_inception_graph.pb"),
                   ModelHelpers.GetAssetsPath("model", "imageClassifier.zip"));
                await modelBuilder.BuildAndTrain();

                var modelEvaluator = new ModelEvaluator(
                    ModelHelpers.GetAssetsPath("data", "tags.tsv"),
                    ModelHelpers.GetAssetsPath("images"),
                    ModelHelpers.GetAssetsPath("model", "imageClassifier.zip"));
                await modelEvaluator.Evaluate();
            } catch (Exception ex)
            {

            }
        }
    }
}
