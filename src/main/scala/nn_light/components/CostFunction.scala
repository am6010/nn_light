package nn_light.components

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.log

trait CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double]): Double
}

class EntropyCostFunction() extends CostFunction {
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    val ones= DenseMatrix.ones[Double](y.rows, y.cols)
    val m = y.cols
    sum((y *:* log(aL)) + ((ones - y) *:* log(ones - aL))) / m 
  }
}

object EntropyCostFunction {
  def apply(): EntropyCostFunction = new EntropyCostFunction()
}
