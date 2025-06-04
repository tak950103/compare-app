using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WpfCompareApp.Models
{
    public class ComparisonResult
    {
        public int Row { get; set; }
        public int Column { get; set; }
        public string HandwrittenLabel { get; set; }
        public string PrintedLabel { get; set; }

        public bool IsMatch => HandwrittenLabel == PrintedLabel;

        public override string ToString()
        {
            return $"行 {Row + 1}, 列 {Column + 1}: " +
                   (IsMatch ? $"一致 ✅（{HandwrittenLabel}）" : $"❌ 手書き = {HandwrittenLabel} ／ 印字 = {PrintedLabel}");
        }
    }
}
