using Emgu.CV.CvEnum;
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
    public static class OnnxClassifier
    {
        private static InferenceSession session;
        private static List<string> labels;
        private static readonly int ImageWidth = 32;
        private static readonly int ImageHeight = 20;

        public static void Initialize(string modelPath = "model.onnx", string labelPath = "labels.txt")
        {
            if (session == null)
            {
                session = new InferenceSession(modelPath);
                labels = File.ReadAllLines(labelPath).ToList();
            }
        }

        public static string Predict(Mat mat)
        {
            if (session == null || labels == null)
                throw new InvalidOperationException("OnnxClassifier is not initialized.");

            // グレースケール＆サイズ統一
            var gray = mat.ToImage<Gray, byte>().Resize(ImageWidth, ImageHeight, Inter.Cubic);

            // Tensor作成：float[1, 1, H, W]
            var input = new DenseTensor<float>(new[] { 1, 1, ImageHeight, ImageWidth });
            for (int y = 0; y < ImageHeight; y++)
            {
                for (int x = 0; x < ImageWidth; x++)
                {
                    input[0, 0, y, x] = gray.Data[y, x, 0] / 255.0f; // 正規化
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };

            // 推論実行
            using var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            int maxIndex = Array.IndexOf(output, output.Max());
            return labels[maxIndex];
        }
    }
}
