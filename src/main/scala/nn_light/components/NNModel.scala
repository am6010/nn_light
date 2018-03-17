package nn_light.components

import breeze.linalg.{DenseMatrix, DenseVector}


trait NNModel {
  
  def train(): Unit
  
  def predict(): Unit
}

class DeepNN(X: DenseMatrix[Double], Y: DenseMatrix[Double], context: NNContext) extends NNModel{
  
  def train(): Unit = ???

  def predict(): Unit = ???
}

object DeepNN {
  def apply(X: DenseMatrix[Double], 
            Y: DenseMatrix[Double], 
            context: NNContext): DeepNN = new DeepNN(X, Y, context)
}
