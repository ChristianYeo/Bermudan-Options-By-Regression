using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Random;
using MathNet.Numerics.LinearAlgebra;

namespace Bermudan_Option
{

    public interface IDiffusionModel
    {
        public int Seed { get; set; }
        public MersenneTwister UniformGenerator { get; set; }

        public Matrix<double>[] Diffusion(Vector<double> exerciceDates, int numberOfPaths, bool useAntithetic);
    }
}
