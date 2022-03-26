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
            var degreePol = 10;
            var dimension = 1;
            var regType = Utilities.MyEnums.RegressionMethods.Longstaff;
            var pricing = new PricingEngine(BlackScholesDiffusion, payoffType, strike, interestRate, regType, degreePol, dimension);
            var exerciceDates = Vector<double>.Build.DenseOfArray(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var numberOfPaths = 100000;
            var price = pricing.BackwardPass(exerciceDates, numberOfPaths, true);
            var priceForward = pricing.ForwardPass(exerciceDates, numberOfPaths, true);

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
            var numberOfPaths = 30000;
            var price = pricing.BackwardPass(exerciceDates, numberOfPaths);
            var priceForward = pricing.ForwardPass(exerciceDates, numberOfPaths);

            Console.WriteLine("Backward Price : {0}            Forward Price : {1}", Math.Round(price, 3), Math.Round(priceForward, 3));
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
    }
}
