using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;

// DelimitedWriter.Write(fileName, transition[i], ";");

namespace Bermudan_Option
{
    public static class Test
    {
        public static void TestLongstaffDim1()
        {
            var initialPrice = 11.0;
            var interestRate = 0.05;
            var volatility = 0.4;
            var dividend = 0.0;
            var BlackScholesDiffusion = new BlackScholesOneDimensional(initialPrice, interestRate, volatility, dividend);
            var strike = 15.0;
            var payoffType = Utilities.MyEnums.OptionType.put;
            var degreePol = 4;
            var dimension = 1;
            var regType = Utilities.MyEnums.RegressionMethods.Longstaff;
            var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate, regType, degreePol, dimension);
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var numberOfPaths = 500000;
            var price = pricing.BackwardPass(exerciceDates, numberOfPaths);
            var priceForward = pricing.ForwardPass(exerciceDates, numberOfPaths);

            Console.WriteLine("Backward Price : {0}            Forward Price : {1}", Math.Round(price, 3), Math.Round(priceForward, 3));    
        }
        public static void TestLongstaffDim2()
        {
            var initialPrices = Vector<double>.Build.DenseOfArray(new double[] { 110.0, 110.0 });
            var interestRate = 0.05;
            var volatilities = Vector<double>.Build.DenseOfArray(new double[] { 0.2, 0.2 });
            var dividends = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.1 });
            var correlMatrix = Matrix<double>.Build.DenseIdentity(initialPrices.Count);
            var BlackScholesDiffusion = new BlackScholesMultidimensional(initialPrices, interestRate, volatilities, dividends, correlMatrix);
            var strike = 100.0;
            var payoffType = Utilities.MyEnums.OptionType.bestOfCall;
            var degreePol = 5;
            var dimension = initialPrices.Count;
            var regType = Utilities.MyEnums.RegressionMethods.Longstaff;
            var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate, regType, degreePol, dimension);
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] {1,2,3,4,5,6,7,8,9 }) / 3.0;
            var numberOfPaths = 50000;
            var price = pricing.BackwardPass(exerciceDates, numberOfPaths);
            var priceForward = pricing.ForwardPass(exerciceDates, numberOfPaths);

            Console.WriteLine("Backward Price : {0}            Forward Price : {1}", Math.Round(price, 3), Math.Round(priceForward, 3));
        }
        public static void TestCarriereDim1()
        {
            var initialPrice = 11.0;
            var interestRate = 0.05;
            var volatility = 0.4;
            var dividend = 0.0;
            var BlackScholesDiffusion = new BlackScholesOneDimensional(initialPrice, interestRate, volatility, dividend);
            var strike = 15.0;
            var payoffType = Utilities.MyEnums.OptionType.put;
            var degreePol = 4;
            var dimension = 1;
            var numberOfKnots = 7;
            var regType = Utilities.MyEnums.RegressionMethods.NonParametric;
            var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate, regType, degreePol, dimension, numberOfKnots);
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var numberOfPaths = 300000;
            var price = pricing.BackwardPass(exerciceDates, numberOfPaths);
            var priceForward = pricing.ForwardPass(exerciceDates, numberOfPaths);

            Console.WriteLine("Backward Price : {0}            Forward Price : {1}", Math.Round(price, 3), Math.Round(priceForward, 3));
        }
        public static void TestQuantifDim1()
        {
            var initialPrice = 11.0;
            var interestRate = 0.05;
            var volatility = 0.4;
            var dividend = 0.0;
            var gridSize = 500;
            var BlackScholesDiffusion = new QuantizedOneDimensionalBlackScholesBuilder(initialPrice, interestRate, volatility, dividend, gridSize);
            var strike = 15.0;
            var payoffType = Utilities.MyEnums.OptionType.put;
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var companionWeights = Utilities.OptimalQuantizationTools.EstimateCompanionWeightDim1(exerciceDates, gridSize);
            var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate);
            var price = pricing.Quantization(exerciceDates, companionWeights.transitionKernel, gridSize);

            Console.WriteLine("Quantif Price : {0}", Math.Round(price, 3));
        }
        public static void TestModifiedLongstaffDim1()
        {
            var initialPrice = 11.0;
            var interestRate = 0.05;
            var volatility = 0.4;
            var dividend = 0.0;
            var BlackScholesDiffusion = new BlackScholesOneDimensional(initialPrice, interestRate, volatility, dividend);
            var strike = 15.0;
            var payoffType = Utilities.MyEnums.OptionType.put;
            var degreePol = 4;
            var dimension = 1;
            var regType = Utilities.MyEnums.RegressionMethods.ModifiedLongstaff;
            var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate, regType, degreePol, dimension);
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var numberOfPaths = 100000;
            var price = pricing.BackwardPass(exerciceDates, numberOfPaths);
            var priceForward = pricing.ForwardPass(exerciceDates, numberOfPaths);

            Console.WriteLine("Backward Price : {0}            Forward Price : {1}", Math.Round(price, 3), Math.Round(priceForward, 3));
        }
        public static void TestQuantifVanilla()
        {
            var initialPrice = 110.0;
            var interestRate = 0.05;
            var volatility = 0.4;
            var dividend = 0.0;
            var gridSize = 1000;
            var BlackScholesDiffusion = new QuantizedOneDimensionalBlackScholesBuilder(initialPrice, interestRate, volatility, dividend, gridSize);
            var strike = 100.0;
            var payoffType = Utilities.MyEnums.OptionType.call;
            var maturity = 1.0;
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] {maturity});
            var payoffBuilder = new PayoffSingleAsset(strike, payoffType);
            var states = BlackScholesDiffusion.Diffusion(exerciceDates, gridSize);
            var discountedPayoffs = payoffBuilder.ComputePayoffValues(states[0]) * Math.Exp(-maturity * interestRate);
            var probabilities = Utilities.LoadGaussianQuantizationProbabilities(gridSize, 1);
            var price = Math.Round(probabilities * discountedPayoffs, 4);

            Console.WriteLine("Quantif Vanilla Price : {0}", price);


        }
        public static void TestCvgQuantifDim1()
        {
            var initialPrice = 11.0;
            var interestRate = 0.05;
            var volatility = 0.4;
            var dividend = 0.0;
            var strike = 15.0;
            var payoffType = Utilities.MyEnums.OptionType.put;
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });

            var gridSizeList = new int[27] {100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,
            1500,2000,2500,3000,3500,4000,4500,5000};

            for(var idx = 0; idx < gridSizeList.Length; ++idx)
            {
                var gridSize = gridSizeList[idx];
                var BlackScholesDiffusion = new QuantizedOneDimensionalBlackScholesBuilder(initialPrice, interestRate, volatility, dividend, gridSize);
                var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate);
                var companionWeights = Utilities.OptimalQuantizationTools.EstimateCompanionWeightDim1(exerciceDates, gridSize);
                var price = pricing.Quantization(exerciceDates, companionWeights.transitionKernel, gridSize);

                Console.WriteLine("{0},{1},", gridSize, Math.Round(price, 3));
            }
        }
    }
}
