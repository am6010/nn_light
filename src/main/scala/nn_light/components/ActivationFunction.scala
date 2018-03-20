package nn_light.components

import breeze.linalg.DenseMatrix
import breeze.numerics.{relu, sigmoid}

trait ActivationFunction {
  def activate(zInput: DenseMatrix[Double]): DenseMatrix[Double]
}

case class Sigmoid() extends ActivationFunction {
  def activate(zInput: DenseMatrix[Double]): DenseMatrix[Double] =  {
    sigmoid(zInput)
  }
}


case class Relu() extends ActivationFunction {
  def activate(zInput: DenseMatrix[Double]): DenseMatrix[Double] = {
   relu(zInput)
  }
}
