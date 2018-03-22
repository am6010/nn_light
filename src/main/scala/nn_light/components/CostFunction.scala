package nn_light.components

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.log

trait CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double], parameters: Parameters): Double
}

class EntropyCostFunction() extends CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double], 
                  parameters: Parameters): Double = {
    val m = y.cols
    if (m == 0) {
      throw new RuntimeException("Inputs array should not be empty")
    }
    val ones= DenseMatrix.ones[Double](y.rows, y.cols)
    - sum((y *:* log(aL)) + ((ones - y) *:* log(ones - aL))) / m 
  }
}

class EntropyCostFunctionL2(lambda: Double) extends CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double], parameters: Parameters): Double = {
    val m = y.cols
    if (m == 0) {
      throw new RuntimeException("Inputs array should not be empty")
    }
    val ones= DenseMatrix.ones[Double](y.rows, y.cols)
    val entropy = - sum((y *:* log(aL)) + ((ones - y) *:* log(ones - aL))) / m
    
    val reg = (lambda / (2 * m)) * parameters.weights.values.foldLeft(0.0){(s, w) => 
      s + sum(w)
    }
    entropy + reg
  }
}

object EntropyCostFunction {
  def apply(): EntropyCostFunction = new EntropyCostFunction()
}
