package nn_light.components

import breeze.linalg.DenseMatrix
import breeze.numerics.{relu, sigmoid}

trait ActivationFunction {
  def activate(zInput: DenseMatrix[Double]): DenseMatrix[Double]
  
  def firstDerivative(zInput: DenseMatrix[Double]): DenseMatrix[Double]
}

case class Sigmoid() extends ActivationFunction {
  def activate(zInput: DenseMatrix[Double]): DenseMatrix[Double] =  {
    sigmoid(zInput)
  }

  def firstDerivative(zInput: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sig = sigmoid(zInput)
    sig *:* (DenseMatrix.ones[Double](zInput.rows, zInput.cols) - sig)
  }
}


case class Relu() extends ActivationFunction {
  def activate(zInput: DenseMatrix[Double]): DenseMatrix[Double] = {
   relu(zInput)
  }

  def firstDerivative(zInput: DenseMatrix[Double]): DenseMatrix[Double] = {
    zInput.map(e => if (e > 0.0) 1.00 else 0.00000001)
  }
}
