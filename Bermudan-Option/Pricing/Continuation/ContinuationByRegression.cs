using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using MathNet.Numerics;

namespace Bermudan_Option
{
    public abstract class ContinuationByRegression : IContinuationEngine
    {
        public List<Vector<double>> regressionCoefficients { get; set; }
        protected RegressionFunctor polynome { get; set; }

        public ContinuationByRegression(int degree, int dimension)
        {
            if(dimension > 2)
            {
                throw new NotImplementedException("Not managed !");
            }
            this.polynome = new RegressionFunctor(degree, dimension);
            regressionCoefficients = new List<Vector<double>>();
        }

        public abstract Matrix<double> ComputeRegressors(Matrix<double> stateVariables);



        public Vector<double> ComputeContinuation(Vector<double> nextValues, Matrix<double> states)
        {
            var regressors = ComputeRegressors(states);
            var regCoeff = LeastSquareRegression(nextValues, regressors);
            regressionCoefficients.Add(regCoeff);

            return regressors.Multiply(regCoeff);
        }

        public Vector<double> LeastSquareRegression(Vector<double> values, Matrix<double> regressionbBasis)
        {
            return regressionbBasis.TransposeThisAndMultiply(regressionbBasis).QR().Solve(regressionbBasis.TransposeThisAndMultiply(values));
        }
        public Vector<double> ComputeContinuationForwardPass(Matrix<double> states, int indexExerciceDate)
        {
            var regressors = ComputeRegressors(states);
            return regressors.Multiply(regressionCoefficients[indexExerciceDate]);
        }
    }

    public class ContinuationByLongstaffSchwartz : ContinuationByRegression
    {
        public ContinuationByLongstaffSchwartz(int degree, int dimension) : base(degree, dimension)
        {
            // LSMC base
        }
        public override Matrix<double> ComputeRegressors(Matrix<double> stateVariables)
        {
            return polynome.Evaluate(stateVariables);
        }
    }

    public class ContinuationByModifiedLongstaffSchwartz : ContinuationByRegression
    {
        public ContinuationByModifiedLongstaffSchwartz(PayoffBuilder payoff, int degree, int dimension): base(degree, dimension)
        {
            var newBasisFunctions = new List<Func<Vector<double>, double>>();
            for(var i = 0; i < polynome.basisFunctions.Length; i++)
            {
                newBasisFunctions.Add(polynome.basisFunctions[i]);
            }
            
            if(payoff is PayoffSingleAsset Spayoff)
            {
                newBasisFunctions.Add(x => Spayoff.payoffFunction(x[0]));
            }

            else if(payoff is PayoffMultiAsset Mpayoff)
            {
                newBasisFunctions.Add(x => Mpayoff.payoffFunction(x));
            }

            this.polynome.NumberOfRegressor += 1;
            this.polynome.basisFunctions = newBasisFunctions.ToArray();
        }
        public override Matrix<double> ComputeRegressors(Matrix<double> stateVariables)
        {
            return polynome.Evaluate(stateVariables);
        }
    }

    public class ContinuationByCarriere : ContinuationByRegression
    {
        private readonly int numberOfKnots;

        public ContinuationByCarriere(int degree, int dimension, int nbKnots) : base(degree, dimension)
        {
            this.numberOfKnots = nbKnots;
        }
        public Func<Vector<double>, double>[] BuildNewBasisFunctions(int degree, Vector<double> listOfKnots)
        {
            var newBasisFunctions = new List<Func<Vector<double>, double>>();

            for (var iKnot = 0; iKnot < listOfKnots.Count; ++iKnot)
            {
                var knot = listOfKnots[iKnot];
                newBasisFunctions.Add(x => Math.Max(Math.Pow(x[0] - knot, degree), 0.0));
            }

            return newBasisFunctions.ToArray();
        }
        public Vector<double> ComputeListOfKnots(int numberOfKnots, Vector<double> values)
        {
            if (numberOfKnots <= 2)
            {
                throw new Exception("error !");
            }

            var result = Vector<double>.Build.Dense(numberOfKnots - 1);
            var valuesCopy = values.Clone();
            var nbElts = values.Count;

            Utilities.SortingAlgorithms.QuickSort(valuesCopy, 0, nbElts - 1);

            for (var i = 0; i < numberOfKnots - 1; i++)
            {
                result[i] = valuesCopy[(i + 1) * nbElts / numberOfKnots];
            }

            return result;
        }
        public override Matrix<double> ComputeRegressors(Matrix<double> stateVariables)
        {
            var listOfKnots = ComputeListOfKnots(numberOfKnots, stateVariables.Column(0));
            var additionalBasis = BuildNewBasisFunctions(polynome.degree, listOfKnots);
            var regressors = polynome.Evaluate(stateVariables, additionalBasis);

            return regressors;
        }
    }




    public class RegressionFunctor
    {
        public Func<Vector<double>, double>[] basisFunctions { get; set; }
        public int NumberOfRegressor { get; set; }
        public int degree { get; }

        public RegressionFunctor(int degree, int numberOfStateVariable)
        {
            this.degree = degree;

            var functorList = new List<Func<Vector<double>, double>>();
            functorList.Add(x => 1);

            for (var degMonome = 1; degMonome <= degree; ++degMonome)
            {
                for (var idx = 0; idx < numberOfStateVariable; ++idx)
                {
                    var index = idx;
                    var exponent = degMonome;
                    functorList.Add(x => Math.Pow(x[index], exponent));
                }
            }

            for (var degPolynome = 2; degPolynome <= degree; ++degPolynome)
            {
                for (var deg1 = 1; deg1 < degPolynome; ++deg1)
                {
                    for (var idx1 = 0; idx1 < numberOfStateVariable; ++idx1)
                    {
                        for (var idx2 = 0; idx2 < idx1; ++idx2)
                        {
                            var exponent1 = deg1;
                            var exponent2 = degPolynome - deg1;
                            var index1 = idx1;
                            var index2 = idx2;
                            functorList.Add(x => Math.Pow(x[index1], exponent1) * Math.Pow(x[index2], exponent2));
                        }
                    }
                }
            }

            basisFunctions = functorList.ToArray();
            NumberOfRegressor = basisFunctions.Length;
        }
        public Matrix<double> Evaluate(Matrix<double> stateVariables)
        {
            var nSim = stateVariables.RowCount;
            var standardizedMatrix = stateVariables; 
            //var standardizedMatrix = StandardizeMatrix(stateVariables);
            var output = Matrix<double>.Build.Dense(nSim, NumberOfRegressor);
            for (var iSim = 0; iSim < nSim; ++iSim)
            {
                for (var iRegressor = 0; iRegressor < NumberOfRegressor; ++iRegressor)
                {
                    output.At(iSim, iRegressor, basisFunctions[iRegressor](standardizedMatrix.Row(iSim)));
                }
            }
            return output;
        }
        public Matrix<double> Evaluate(Matrix<double> stateVariables, Func<Vector<double>, double>[] newBasis)
        {
            var nSim = stateVariables.RowCount;
            var standardizedMatrix = StandardizeMatrix(stateVariables);
            var numberOfStateVariable = stateVariables.ColumnCount;
            var totalNumberOfRegressors = NumberOfRegressor + newBasis.Length;

            var output = Matrix<double>.Build.Dense(nSim, totalNumberOfRegressors);

            for (var iSim = 0; iSim < nSim; ++iSim)
            {
                for (var iRegressor = 0; iRegressor < NumberOfRegressor; ++iRegressor)
                {
                    output.At(iSim, iRegressor, basisFunctions[iRegressor](standardizedMatrix.Row(iSim)));
                }

                for (var iRegressor = NumberOfRegressor; iRegressor < totalNumberOfRegressors; iRegressor++)
                {
                    var indexRegressor = iRegressor - NumberOfRegressor;
                    output.At(iSim, iRegressor, newBasis[indexRegressor](standardizedMatrix.Row(iSim)));
                }
            }

            return output;
        }
        public static Matrix<double> StandardizeMatrix(Matrix<double> inputMatrix)
        {
            var standardizedColumns =
                inputMatrix.EnumerateColumns().Select(x => x - x.Mean()).Select(y => y.StandardDeviation().AlmostEqual(0.0) ? y : y / y.StandardDeviation());
            var standardizedMatrix = Matrix<double>.Build.DenseOfColumnVectors(standardizedColumns);
            return standardizedMatrix;
        }
    }
}
