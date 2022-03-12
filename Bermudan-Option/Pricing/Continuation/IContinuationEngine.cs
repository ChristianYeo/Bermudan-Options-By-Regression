using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace Bermudan_Option
{
    public interface IContinuationEngine
    {
        public Vector<double> ComputeContinuation(Vector<double> nextValues, Matrix<double> states);
        public Vector<double> ComputeContinuationForwardPass(Matrix<double> states, int indexExerciceDate);
    }
}
