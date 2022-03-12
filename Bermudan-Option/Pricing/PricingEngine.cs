using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace Bermudan_Option
{
    public class PricingEngine
    {
        private readonly PayoffBuilder payoffBuilder;
        private readonly IDiffusionModel diffusionModel;
        private readonly IContinuationEngine continuationEngine;
        private readonly double interestRate;
        public PricingEngine(IDiffusionModel diffusionModel, Utilities.MyEnums.OptionType optionType, double strike, double interestRate,
            Utilities.MyEnums.RegressionMethods rm, int polynomeDegree, int dimension, int numberOfKnots = 0)
        {
            if(optionType == Utilities.MyEnums.OptionType.call || optionType == Utilities.MyEnums.OptionType.put)
            {
                this.payoffBuilder = new PayoffSingleAsset(strike, optionType);
            }

            else
            {
                this.payoffBuilder = new PayoffMultiAsset(strike, optionType);
            }

            this.diffusionModel = diffusionModel;

            if(rm == Utilities.MyEnums.RegressionMethods.Longstaff)
            {
                this.continuationEngine = new ContinuationByLongstaffSchwartz(polynomeDegree, dimension);
            }

            else if(rm == Utilities.MyEnums.RegressionMethods.ModifiedLongstaff)
            {
                this.continuationEngine = new ContinuationByModifiedLongstaffSchwartz(payoffBuilder, polynomeDegree, dimension);
            }

            else // non parametric regression ==> carriere
            {
                this.continuationEngine = new ContinuationByCarriere(polynomeDegree, dimension, numberOfKnots);
            }

            this.interestRate = interestRate;
        }
        public PricingEngine(IDiffusionModel diffusionModel, Utilities.MyEnums.OptionType optionType, double strike, double interestRate)
        {
            if (optionType == Utilities.MyEnums.OptionType.call || optionType == Utilities.MyEnums.OptionType.put)
            {
                this.payoffBuilder = new PayoffSingleAsset(strike, optionType);
            }

            else
            {
                this.payoffBuilder = new PayoffMultiAsset(strike, optionType);
            }

            this.diffusionModel = diffusionModel;
            this.continuationEngine = null;
            this.interestRate = interestRate;
        }
        public double BackwardPass(Vector<double> exerciceDates, int numberOfPaths)
        {
            // paths generation
            var indexLastExerciceDate = exerciceDates.Count - 1;
            var states = diffusionModel.Diffusion(exerciceDates, numberOfPaths);

            // final condition
            var thisExerciceDate = exerciceDates[indexLastExerciceDate];
            var discountFactor = Math.Exp(-interestRate * thisExerciceDate);
            var nextValues = payoffBuilder.ComputePayoffValues(states[indexLastExerciceDate]) * discountFactor;

            // backward induction

            for(var iExerciceDate = indexLastExerciceDate - 1; iExerciceDate >= 0; iExerciceDate--)
            {
                thisExerciceDate = exerciceDates[iExerciceDate];
                discountFactor = Math.Exp(-interestRate * thisExerciceDate);
                var discountedPayoff = payoffBuilder.ComputePayoffValues(states[iExerciceDate]) * discountFactor;
                var continuationValue = continuationEngine.ComputeContinuation(nextValues, states[iExerciceDate]);

                for(var iPath = 0; iPath < numberOfPaths; iPath++)
                {
                    var payoff = discountedPayoff[iPath];
                    if (payoff >= continuationValue[iPath])
                    {
                        nextValues[iPath] = payoff;
                    }
                }
            }

            return nextValues.Average();
        }
        public double ForwardPass(Vector<double> exerciceDates, int numberOfPaths)
        {
            // new paths generation
            var states = diffusionModel.Diffusion(exerciceDates, numberOfPaths);
            var numberOfExerciceDates = exerciceDates.Count;
            var optimalValues = Vector<double>.Build.Dense(numberOfPaths);
            var indexLastExerciceDate = exerciceDates.Count - 1;
            var earlyExercise = new bool[numberOfPaths];
            var discountFactor = 0.0;

            // forward valorisation

            for (var iExerciceDate = 0; iExerciceDate < numberOfExerciceDates - 1; ++iExerciceDate)
            {
                var thisExerciceDate = exerciceDates[iExerciceDate];
                discountFactor = Math.Exp(-thisExerciceDate * interestRate);
                var thisPayoffs = payoffBuilder.ComputePayoffValues(states[iExerciceDate]) * discountFactor;
                var continuationValues = continuationEngine.ComputeContinuationForwardPass(states[iExerciceDate], indexLastExerciceDate - iExerciceDate - 1);
    
                for (var iPath = 0; iPath < numberOfPaths; ++iPath)
                {
                    var payoff = thisPayoffs[iPath];
                    if (payoff >= continuationValues[iPath] && !earlyExercise[iPath])
                    {
                        optimalValues[iPath] = payoff;
                        earlyExercise[iPath] = true;
                    }
                }
            }

            discountFactor = Math.Exp(-exerciceDates[indexLastExerciceDate] * interestRate);
            var discountedPayoffs = payoffBuilder.ComputePayoffValues(states[indexLastExerciceDate]) * discountFactor;

            for (var iPath = 0; iPath < numberOfPaths; ++iPath)
            {
                if (!earlyExercise[iPath])
                {
                    optimalValues[iPath] = discountedPayoffs[iPath];
                }
            }

            return optimalValues.Average();
        }
    }
}
