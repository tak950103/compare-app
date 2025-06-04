using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using System.Windows.Media.Imaging;
using System.Drawing.Imaging;

namespace WpfCompareApp.Helpers
{
    public static class ImageProcessor
    {
        /// <summary>
        /// 画像を25行に分割する
        /// </summary>
        public static List<Mat> SplitIntoRows(string imagePath, int expectedRowCount = 25)
        {
            var image = new Image<Bgr, byte>(imagePath);
            var gray = image.Convert<Gray, byte>();

            // ノイズ除去
            CvInvoke.GaussianBlur(gray, gray, new Size(3, 3), 0);

            // 二値化
            var binary = new Mat();
            CvInvoke.Threshold(gray, binary, 0, 255, ThresholdType.BinaryInv | ThresholdType.Otsu);

            // 水平の輪郭検出 or モルフォロジーで行領域抽出
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(1, 10), new Point(-1, -1));
            var morphed = new Mat();
            CvInvoke.MorphologyEx(binary, morphed, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Reflect, new MCvScalar());

            // 輪郭抽出
            var contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(morphed, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            List<Rectangle> rowRects = new List<Rectangle>();
            for (int i = 0; i < contours.Size; i++)
            {
                var rect = CvInvoke.BoundingRectangle(contours[i]);
                if (rect.Height > 20 && rect.Width > image.Width / 2) // 横長の領域のみ
                {
                    rowRects.Add(rect);
                }
            }

            // 上から下に並び替え
            rowRects.Sort((a, b) => a.Top.CompareTo(b.Top));

            // 過不足があれば、等間隔分割にフォールバック
            if (rowRects.Count < expectedRowCount)
            {
                rowRects.Clear();
                int rowHeight = image.Height / expectedRowCount;
                for (int i = 0; i < expectedRowCount; i++)
                {
                    rowRects.Add(new Rectangle(0, i * rowHeight, image.Width, rowHeight));
                }
            }

            // 各行をMatとして切り出す
            var rowImages = new List<Mat>();
            foreach (var rect in rowRects)
            {
                rowImages.Add(new Mat(image.Mat, rect));
            }

            return rowImages;
        }

        public static BitmapImage ToBitmapImage(this Mat mat)
        {
            using (var ms = new MemoryStream())
            {
                // Mat → Image<Bgr, Byte> → Bitmap
                var bmp = mat.ToImage<Bgr, byte>().ToBitmap();
                bmp.Save(ms, ImageFormat.Png);
                ms.Seek(0, SeekOrigin.Begin);

                var image = new BitmapImage();
                image.BeginInit();
                image.CacheOption = BitmapCacheOption.OnLoad;
                image.StreamSource = ms;
                image.EndInit();
                image.Freeze();

                return image;
            }
        }

        public static Mat[][] SplitIntoCellsWithOffsets(
            Mat sourceImage,
            int[] startYList,      // 各行の開始Y座標
            int[] rowHeights,      // 各行の高さ
            int[] startXList,      // 各列の開始X座標
            int[] columnWidths     // 各列の幅
        )
        {
            int rows = startYList.Length;
            int cols = startXList.Length;
            var result = new Mat[rows][];

            for (int row = 0; row < rows; row++)
            {
                result[row] = new Mat[cols];

                for (int col = 0; col < cols; col++)
                {
                    int x = startXList[col];
                    int y = startYList[row];
                    int w = columnWidths[col];
                    int h = rowHeights[row];

                    // 範囲チェック
                    if (x + w > sourceImage.Width) w = sourceImage.Width - x;
                    if (y + h > sourceImage.Height) h = sourceImage.Height - y;
                    if (w > 0 && h > 0)
                    {
                        Rectangle roi = new Rectangle(x, y, w, h);
                        result[row][col] = new Mat(sourceImage, roi);
                    }
                    else
                    {
                        result[row][col] = new Mat(); // 空マットで代用
                    }
                }
            }

            return result;
        }
    }
}
