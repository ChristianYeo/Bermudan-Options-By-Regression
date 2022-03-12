using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Random;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using IronXL;

namespace Bermudan_Option
{
    public static class Utilities
    {
        public static double BoxMullerDim1(MersenneTwister uniform)
        {
            var u1 = 1.0 - uniform.NextDouble();
            var u2 = 1.0 - uniform.NextDouble();

            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
        public static Vector<double> BoxMullerDim2(MersenneTwister uniform)
        {
            var u1 = 1.0 - uniform.NextDouble();
            var u2 = 1.0 - uniform.NextDouble();

            var gaussian = Vector<double>.Build.Dense(2);

            gaussian[0] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            gaussian[1] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            return gaussian;
        }
        public static Vector<double> GenerateGaussians(MersenneTwister uniform, int numberOfGaussians)
        {
            var gaussians = Vector<double>.Build.Dense(numberOfGaussians);
            var a = numberOfGaussians % 2;
            var end = (a == 0) ? (numberOfGaussians - 1) : (numberOfGaussians - 2);

            var i = 0;

            for (; i < end; i += 2)
            {
                var gauss = BoxMullerDim2(uniform);
                gaussians[i] = gauss[0];
                gaussians[i + 1] = gauss[1];
            }

            for (var j = 0; j < a; ++j)
            {
                gaussians[i + j] = BoxMullerDim1(uniform);
            }

            return gaussians;
        }
        public static Matrix<double> ComputeMatrixSquareRoot(Matrix<double> inputMatrix)
        {
            var size = inputMatrix.RowCount;
            var diagVector = Vector<double>.Build.Dense(size);
            var lMatrix = Matrix<double>.Build.Dense(size, size);
            for (var j = 0; j < size; j++)
            {
                var diagValue = inputMatrix[j, j];
                for (var k = 0; k < j; k++)
                {
                    diagValue -= lMatrix[j, k] * lMatrix[j, k] * diagVector[k];
                }
                diagVector.At(j, diagValue);
                lMatrix.At(j, j, 1.0);
                for (var i = j + 1; i < size; ++i)
                {
                    var lValue = inputMatrix[i, j];
                    for (var k = 0; k < j; k++)
                    {
                        lValue -= lMatrix[i, k] * lMatrix[j, k] * diagVector[k];
                    }
                    lValue /= diagValue;
                    lMatrix.At(i, j, lValue);
                }
            }
            var sqrtDiagMatrix = Matrix<double>.Build.Diagonal(diagVector.Select(x => x > 0.0 ? Math.Sqrt(x) : 0.0).ToArray());
            var output = Matrix<double>.Build.Dense(size, size);
            lMatrix.Multiply(sqrtDiagMatrix, output);
            return output;
        }
        public static Matrix<double>[] GenerateCorrelatedBM(Matrix<double> choleskyMatrix, Vector<double> diffusionDates, MersenneTwister uniformGenerator, 
            int numberOfPaths)
        {
            var numberOfDates = diffusionDates.Count;
            var dim = choleskyMatrix.RowCount;
            var BM = new Matrix<double>[numberOfDates];

            var startDate = 0.0;

            for (var iDate = 0; iDate < numberOfDates; ++iDate)
            {
                var thisDate = diffusionDates[iDate];
                var sqrtDeltaTime = Math.Sqrt(thisDate - startDate);
                BM[iDate] = Matrix<double>.Build.Dense(numberOfPaths, dim);

                for (var iPath = 0; iPath < numberOfPaths; ++iPath)
                {
                    BM[iDate].SetRow(iPath, sqrtDeltaTime * choleskyMatrix.Multiply(GenerateGaussians(uniformGenerator, dim)));
                }

                startDate = thisDate;
            }

            for (var iDate = 1; iDate < numberOfDates; ++iDate)
            {
                BM[iDate] += BM[iDate - 1];
            }

            return BM;
        }
        public static class SortingAlgorithms
        {
            public static void Swap(ref Vector<double> arr, int i, int j)
            {
                var temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
            public static int FindPartitionForQuickSort(Vector<double> arr, int low, int high)
            {
                var pivot = arr[high];
                var i = (low - 1);

                for (var j = low; j <= high - 1; j++)
                {
                    if (arr[j] < pivot)
                    {
                        i++;
                        Swap(ref arr, i, j);
                    }
                }
                Swap(ref arr, i + 1, high);
                return (i + 1);
            }
            public static void QuickSort(Vector<double> arr, int low, int high)
            {
                if (low < high)
                {
                    int pi = FindPartitionForQuickSort(arr, low, high);
                    QuickSort(arr, low, pi - 1);
                    QuickSort(arr, pi + 1, high);
                }
            }
        }
        public static class MyEnums
        {
            public enum RegressionMethods
            {
                Longstaff = 0,
                NonParametric = 1,
                ModifiedLongstaff = 2
            }
            public enum OptionType
            {
                // cas mono
                call = 0,
                put = 1,
                // cas multi
                bestOfCall = 2,
                worstOfPut = 3,
                geometricCall = 4,
                geometricPut = 5
            };
            public enum OptimalStoppingTimePricingMethods
            {
                Longstaff = 0,
                Carriere = 1, // non parametric regression
            }
        }
        public static class SearchAlgorithms
        {
            public static int BinarySearchLowIndex<T>(T[] tab, T elt)where T : System.IComparable<T>
            {
                var low = 0;
                var high = tab.Length;

                if (tab.Length > 0 && elt.CompareTo(tab[tab.Length - 1]) >= 0)
                {
                    return tab.Length - 1;
                }

                while (low <= high)
                {
                    var mid = low + (high - low) / 2;
                    if (tab[mid].CompareTo(elt) <= 0 && tab[mid + 1].CompareTo(elt) > 0)
                    {
                        return mid;
                    }
                    else if (tab[mid].CompareTo(elt) < 0)
                    {
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid - 1;
                    }
                }
                return -1;
            }
            public static int BinarySearchIndex<T>(T[] tab, T elt) where T : System.IComparable<T>
            {
                var low = 0;
                var high = tab.Length - 1;

                while (low <= high)
                {
                    var mid = low + (high - low) / 2;
                    if (tab[mid].CompareTo(elt) == 0)
                    {
                        return ++mid;
                    }
                    if (tab[mid].CompareTo(elt) < 0)
                    {
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid - 1;
                    }
                }
                return -1;
            }
            public static (double minimum, double maximum) FindMinMaxInArray(Vector<double> values)
            {
                var mid = values.Count / 2;
                var size = values.Count;
                for (var idx = 0; idx < mid; ++idx)
                {
                    if (values[2 * mid] > values[2 * mid + 1])
                    {
                        var temp = values[2 * mid];
                        values[2 * mid] = values[2 * mid + 1];
                        values[2 * mid + 1] = temp;
                    }
                }
                var maximum = values[1];
                for (var idx = 1; idx < mid; ++idx)
                {
                    if (values[2 * idx + 1] > maximum)
                    {
                        maximum = values[2 * idx + 1];
                    }
                }
                var minimum = values[0];
                for (var idx = 0; idx < mid; ++idx)
                {
                    if (values[2 * idx] < minimum)
                    {
                        minimum = values[2 * idx];
                    }
                }
                if (values.Count % 2 != 0)
                {
                    if (values[size - 1] > maximum)
                    {
                        maximum = values[size - 1];
                    }
                    else if (values[size - 1] < minimum)
                    {
                        minimum = values[size - 1];
                    }
                }
                return (minimum, maximum);
            }
        }
    }
}
