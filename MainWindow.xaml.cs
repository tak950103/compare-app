using Emgu.CV.CvEnum;
using Emgu.CV;
using Microsoft.Win32;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using WpfCompareApp.Helpers;
using WpfCompareApp.Models;

namespace WpfCompareApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string handwrittenImagePath;
        private string printedImagePath;
        private Mat[][] handwrittenCells;
        private Mat[][] printedCells;
        private List<ComparisonResult> comparisonResults = new();

        public MainWindow()
        {
            InitializeComponent();
        }

        // 手書き画像選択
        private void SelectHandwrittenImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "画像ファイル (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg"
            };

            if (dlg.ShowDialog() == true)
            {
                handwrittenImagePath = dlg.FileName;
                HandwrittenImage.Source = new BitmapImage(new Uri(handwrittenImagePath));
            }
        }

        // 印字画像選択
        private void SelectPrintedImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "画像ファイル (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg"
            };

            if (dlg.ShowDialog() == true)
            {
                printedImagePath = dlg.FileName;
                PrintedImage.Source = new BitmapImage(new Uri(printedImagePath));
            }
        }

        // 比較実行ボタン
        private void CompareImages_Click(object sender, RoutedEventArgs e)
        {
            ComparisonResults.Items.Clear();
            OnnxClassifier.Initialize();
            PrintedClassifier.Initialize();

            if (string.IsNullOrEmpty(handwrittenImagePath) || string.IsNullOrEmpty(printedImagePath))
            {
                MessageBox.Show("両方の画像を選択してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // セルの行・列数（必要に応じて調整）
            int rowCount = 25;
            int colCount = 10;

            // 手書き画像と印字画像のセル分割
            var handwrittenMat = new Mat(handwrittenImagePath, ImreadModes.Color);
            var printedMat = new Mat(printedImagePath, ImreadModes.Color);

            int[] startXList = new int[]
            {
                2, 37, 72, 107, 142, 177, 212, 247, // 狭い8列（幅32ずつ）
                282, 352, 421, 490                // 広い4列（幅63ずつ）
            };

            int[] columnWidths = new int[]
            {
                32, 32, 32, 32, 32, 32, 32, 32,
                63, 63, 63, 63
            };

            int[] rowHeights = Enumerable.Repeat(20, 25).ToArray();

            int[] startYList = new int[]
            {
                 3, 29, 55, 81, 107, 133, 159, 185, 211, 237,
                 263, 289, 315, 341, 367, 393, 419, 445, 471, 499,
                 526, 552, 578, 605, 631
            };

            handwrittenCells = ImageProcessor.SplitIntoCellsWithOffsets(
                handwrittenMat, startYList, rowHeights, startXList, columnWidths);

            printedCells = ImageProcessor.SplitIntoCellsWithOffsets(
                printedMat, startYList, rowHeights, startXList, columnWidths);

            // 比較処理
            comparisonResults.Clear();

            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnWidths.Length; j++)
                {
                    string hwLabel = OnnxClassifier.Predict(handwrittenCells[i][j]);
                    string prLabel = PrintedClassifier.Predict(printedCells[i][j]);

                    comparisonResults.Add(new ComparisonResult
                    {
                        Row = i,
                        Column = j,
                        HandwrittenLabel = hwLabel,
                        PrintedLabel = prLabel
                    });
                }
            }

            ComparisonResults.ItemsSource = comparisonResults;

            RowPreviewImage.Source = handwrittenCells[0][0].ToBitmapImage();
            PrintedPreviewImage.Source = printedCells[0][0].ToBitmapImage();

            // 集計して表示
            int matchCount = comparisonResults.Count(r => r.HandwrittenLabel == r.PrintedLabel);
            int mismatchCount = comparisonResults.Count - matchCount;

            MatchCountText.Text = matchCount.ToString();
            MismatchCountText.Text = mismatchCount.ToString();
        }

        private void ComparisonResults_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComparisonResults.SelectedItem is ComparisonResult selected &&
                handwrittenCells != null && printedCells != null &&
                selected.Row < handwrittenCells.Length &&
                selected.Column < handwrittenCells[selected.Row].Length &&
                handwrittenCells[selected.Row][selected.Column] != null)
            {
                RowPreviewImage.Source = handwrittenCells[selected.Row][selected.Column].ToBitmapImage();
                PrintedPreviewImage.Source = printedCells[selected.Row][selected.Column].ToBitmapImage();
            }
        }


    }
}