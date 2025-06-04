using Emgu.CV.Structure;
using Emgu.CV;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace WpfCompareApp.Helpers
{
    public static class PrintedClassifier
    {
        private static InferenceSession session;
        private static string[] labels;

        public static void Initialize()
        {
            if (session == null)
            {
                session = new InferenceSession("model_print.onnx");

                labels = File.ReadAllLines("labels_print.txt");
            }
        }

        public static string Predict(Mat image)
        {
            var resized = image.ToImage<Gray, byte>().Resize(32, 20, Emgu.CV.CvEnum.Inter.Cubic);
            float[] inputData = resized.Data.Cast<byte>().Select(v => v / 255f).ToArray();

            var tensor = new DenseTensor<float>(inputData, new[] { 1, 1, 20, 32 }); // チャンネル, H, W
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            int predicted = Array.IndexOf(output, output.Max());
            return labels[predicted];
        }
    }
}
