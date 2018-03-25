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
    val entropy = - sum((y *:* log(aL)) + ((1.0 - y) *:* log(1.0 - aL))) / m
    if (Double.NaN.equals(entropy)) 0.0
    else entropy
  }
}

object EntropyCostFunction {
  def apply(): EntropyCostFunction = new EntropyCostFunction()
}

class EntropyCostFunctionL2(lambda: Double) extends CostFunction {
  
  private val entropyFun = new EntropyCostFunction()
  
  def computeCost(aL: DenseMatrix[Double], y: DenseMatrix[Double], parameters: Parameters): Double = {
    val m = y.cols
    if (m == 0) {
      throw new RuntimeException("Inputs array should not be empty")
    }
    
    val entropy = entropyFun.computeCost(aL, y, parameters)
    
    val reg = (lambda / (2 * m)) * parameters.weights.values.foldLeft(0.0){(s, w) =>
      s + sum(w *:* w)
    }
    entropy + reg
  }
}

object EntropyCostFunctionL2 {
  def apply(lambda: Double): EntropyCostFunctionL2 = new EntropyCostFunctionL2(lambda)
}
