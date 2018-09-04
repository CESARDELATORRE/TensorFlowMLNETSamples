using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace TensorFlowSharpInceptionv3Scoring
{
    public class AppSettings
    {
        public string AIModelsPath { get; set; }
        public string TensorFlowPredictionDefaultModel { get; set; } //TensorFlowPreTrained|TensorFlowCustom
    }
}
