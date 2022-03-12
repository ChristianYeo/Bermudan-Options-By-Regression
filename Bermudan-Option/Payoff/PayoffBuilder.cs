using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace Bermudan_Option
{
    public abstract class PayoffBuilder
    {
        protected double strike;
        protected Utilities.MyEnums.OptionType optionType;
        public PayoffBuilder(double strike, Utilities.MyEnums.OptionType optionType)
        {
            this.strike = strike;
            this.optionType = optionType;
        }
        public abstract Vector<double> ComputePayoffValues(Matrix<double> assetPrices);
    }

    public class PayoffSingleAsset : PayoffBuilder
    {
        public Func<double, double> payoffFunction { get; }
        public PayoffSingleAsset(double strike, Utilities.MyEnums.OptionType optionType) : base(strike, optionType)
        {
            this.strike = strike;
            this.optionType = optionType;

            switch (this.optionType)
            {
                case Utilities.MyEnums.OptionType.call:
                    payoffFunction = x => Math.Max(x - this.strike, 0.0);
                    break;

                case Utilities.MyEnums.OptionType.put:
                    payoffFunction = x => Math.Max(this.strike - x, 0.0);
                    break;

                default:
                    payoffFunction = x => 0.0;
                    break;
            }
        }
        public override Vector<double> ComputePayoffValues(Matrix<double> assetPrices)
        {
            var payoffs = Vector<double>.Build.DenseOfEnumerable(assetPrices.Column(0).Select(x => payoffFunction(x)));

            return payoffs;
        }
    }

    public class PayoffMultiAsset : PayoffBuilder
    {
        public Func<Vector<double>, double> payoffFunction { get; }
        public PayoffMultiAsset(double strike, Utilities.MyEnums.OptionType optionType) : base(strike, optionType)
        {
            this.strike = strike;
            this.optionType = optionType;

            switch (this.optionType)
            {
                case Utilities.MyEnums.OptionType.bestOfCall:
                    payoffFunction = x => Math.Max(x.Maximum() - this.strike, 0.0);
                    break;

                case Utilities.MyEnums.OptionType.worstOfPut:
                    payoffFunction = x => Math.Max(this.strike - x.Minimum(), 0.0);
                    break;

                case Utilities.MyEnums.OptionType.geometricCall:
                    payoffFunction = x => Math.Max(x.GeometricMean() - this.strike, 0.0);
                    break;

                case Utilities.MyEnums.OptionType.geometricPut:
                    payoffFunction = x => Math.Max(this.strike - x.GeometricMean(), 0.0);
                    break;

                default:
                    payoffFunction = x => 0.0;
                    break;
            }
        }
        public override Vector<double> ComputePayoffValues(Matrix<double> assetPrices)
        {
            var payoffs = Vector<double>.Build.DenseOfEnumerable(assetPrices.EnumerateRows().Select(x => payoffFunction(x)));

            return payoffs;
        }
    }
}
