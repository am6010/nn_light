package nn_light.components

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.log

trait CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double]): Double
}

class EntropyCostFunction() extends CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    val m = y.cols
    if (m == 0) {
      throw new RuntimeException("Inputs array should not be empty")
    }
    val ones= DenseMatrix.ones[Double](y.rows, y.cols)
    sum((y *:* log(aL)) + ((ones - y) *:* log(ones - aL))) / m 
  }
}

object EntropyCostFunction {
  def apply(): EntropyCostFunction = new EntropyCostFunction()
}
