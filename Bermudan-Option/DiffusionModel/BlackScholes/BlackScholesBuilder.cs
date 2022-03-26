using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Random;
using MathNet.Numerics.LinearAlgebra;

namespace Bermudan_Option
{
    public abstract class BSOneDimensional : IDiffusionModel
    {
        protected readonly double initialPrice;
        protected readonly double interestRate;
        protected readonly double volatility;
        protected readonly double dividend;

        private readonly int seed;
        protected readonly MersenneTwister uniformGen;
        public int Seed
        {
            get
            {
                return seed;
            }
            set
            {
                Seed = value;
            }
        }
        public MersenneTwister UniformGenerator
        {
            get
            {
                return uniformGen;
            }
            set
            {
                UniformGenerator = value;
            }
        }

        public BSOneDimensional(double initialPrice, double interestRate, double volatility, double dividend, int seed = 1234)
        {
            this.initialPrice = initialPrice;
            this.interestRate = interestRate;
            this.volatility = volatility;
            this.dividend = dividend;
            this.seed = seed;
            this.uniformGen = new MersenneTwister(seed);
        }
        public abstract Matrix<double>[] Diffusion(Vector<double> exerciceDates, int numberOfPaths, bool useAntithetic);
    }
    public class BlackScholesOneDimensional : BSOneDimensional
    {
        public BlackScholesOneDimensional(double initialPrice, double interestRate, double volatility, double dividend, int seed = 1234) :
            base(initialPrice, interestRate, volatility, dividend, seed)
        {

        }
        public override Matrix<double>[] Diffusion(Vector<double> exerciceDates, int numberOfPaths, bool useAntithetic)
        {
            var numberOfExerciceDates = exerciceDates.Count;
            var prices = new Matrix<double>[numberOfExerciceDates + 1];
            var antitheticCoeff = useAntithetic ? 2 : 1;
            var actualNumberOfPaths = antitheticCoeff * numberOfPaths;

            for (var iExerciceDate = 0; iExerciceDate <= numberOfExerciceDates; ++iExerciceDate)
            {
                prices[iExerciceDate] = Matrix<double>.Build.Dense(actualNumberOfPaths, 1);
            }

            for(var iPath = 0; iPath < actualNumberOfPaths; ++iPath)
            {
                prices[0].At(iPath, 0, initialPrice);
            }

            var startDate = 0.0;
            var coeff = interestRate - dividend - 0.5 * volatility * volatility;

            for (var iExerciceDate = 0; iExerciceDate < numberOfExerciceDates; ++iExerciceDate)
            {
                var thisExerciceDate = exerciceDates[iExerciceDate];
                var deltaTime = thisExerciceDate - startDate;
                var driftPart = coeff * deltaTime;
                var volPart = volatility * Math.Sqrt(deltaTime);
                var gaussiansPlusOrNotAntithetic = Vector<double>.Build.Dense(antitheticCoeff * numberOfPaths);
                var gaussians = Utilities.GenerateGaussians(uniformGen, numberOfPaths);
                gaussiansPlusOrNotAntithetic.SetSubVector(0, numberOfPaths, gaussians);

                if(useAntithetic)
                {
                    gaussiansPlusOrNotAntithetic.SetSubVector(numberOfPaths, numberOfPaths, -gaussians);
                }

                prices[iExerciceDate + 1].SetColumn(0, prices[iExerciceDate].Column(0).PointwiseMultiply((driftPart + volPart * gaussiansPlusOrNotAntithetic).PointwiseExp()));

                startDate = thisExerciceDate;
            }

            return prices.Skip(1).ToArray();
        }
    }
    public abstract class BSMultidimensional : IDiffusionModel
    {
        protected readonly Vector<double> initialPrices;
        protected readonly double interestRate;
        protected readonly Vector<double> volatilities;
        protected readonly Vector<double> dividends;
        protected readonly Matrix<double> correlationMatrix;
        protected readonly Matrix<double> choleskyMatrix;

        private readonly int seed;
        protected readonly MersenneTwister uniformGen;

        public int Seed
        {
            get
            {
                return seed;
            }
            set
            {
                Seed = value;
            }
        }
        public MersenneTwister UniformGenerator
        {
            get
            {
                return uniformGen;
            }
            set
            {
                UniformGenerator = value;
            }
        }

        public BSMultidimensional(Vector<double> initialPrices, double interestRate, Vector<double> volatilities, Vector<double> dividends,
            Matrix<double> correlationMatrix, int seed = 1234)
        {
            this.initialPrices = initialPrices;
            this.interestRate = interestRate;
            this.volatilities = volatilities;
            this.dividends = dividends;
            this.correlationMatrix = correlationMatrix;
            this.seed = seed;
            uniformGen = new MersenneTwister(seed);

            this.choleskyMatrix = Utilities.ComputeMatrixSquareRoot(
                Matrix<double>.Build.DenseOfDiagonalVector(this.volatilities).Multiply(this.correlationMatrix).Multiply(
                Matrix<double>.Build.DenseOfDiagonalVector(this.volatilities)));
        }
        public abstract Matrix<double>[] Diffusion(Vector<double> exerciceDates, int numberOfPaths, bool useAntithetic);
    }

    public class BlackScholesMultidimensional : BSMultidimensional
    {
        public BlackScholesMultidimensional(Vector<double> initialPrices, double interestRate, Vector<double> volatilities, Vector<double> dividends,
            Matrix<double> correlationMatrix, int seed = 1234) : base(initialPrices, interestRate, volatilities, dividends, correlationMatrix, seed)
        {

        }
        public override Matrix<double>[] Diffusion(Vector<double> exerciceDates, int numberOfPaths, bool useAntithetic)
        {
            var dim = initialPrices.Count;
            var numberOfExerciceDates = exerciceDates.Count;
            var prices = new Matrix<double>[numberOfExerciceDates];
            var antitheticCoeff = useAntithetic ? 2 : 1;
            var actualNumberOfPaths = antitheticCoeff * numberOfPaths;

            for (var iExerciceDate = 0; iExerciceDate < numberOfExerciceDates; ++iExerciceDate)
            {
                prices[iExerciceDate] = Matrix<double>.Build.Dense(actualNumberOfPaths, dim);
            }

            var correlatedBM = Utilities.GenerateCorrelatedBM(choleskyMatrix, exerciceDates, uniformGen, actualNumberOfPaths);

            for (var iExerciceDate = 0; iExerciceDate < numberOfExerciceDates; ++iExerciceDate)
            {
                var thisExerciceDate = exerciceDates[iExerciceDate];
                var driftPart = (interestRate - dividends - 0.5 * volatilities.PointwiseMultiply(volatilities)) * thisExerciceDate;

                for (var iAsset = 0; iAsset < dim; ++iAsset)
                {
                    prices[iExerciceDate].SetColumn(iAsset, initialPrices[iAsset] * (driftPart[iAsset] + 
                        correlatedBM[iExerciceDate].Column(iAsset)).PointwiseExp());
                }
            }

            return prices;
        }
    }
}
