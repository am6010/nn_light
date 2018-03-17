package nn_light.components

import breeze.linalg.{*, DenseMatrix, DenseVector}

trait ForwardActivation {
  
  def linearForward(activations: DenseMatrix[Double], 
                    weights: DenseMatrix[Double], 
                    bias: DenseVector[Double]): (DenseMatrix[Double], LinearCache)
  
  def linearForwardActivation(activationsPrev: DenseMatrix[Double], 
                              weights: DenseMatrix[Double],
                              bias: DenseVector[Double],
                              activation: ActivationFunction)
  : (DenseMatrix[Double], (LinearCache, ActivationCache))
  
  def lModelForward(XInput: DenseMatrix[Double], parameters: Parameters)
  : (DenseMatrix[Double], Cache)
}


class LForwardModel() extends ForwardActivation {
  
  def linearForward(activations: DenseMatrix[Double], 
                    weights: DenseMatrix[Double],
                    bias: DenseVector[Double]): (DenseMatrix[Double], LinearCache) = {
    
    val WA = weights * activations
    val z = WA(::, *) + bias
    (z, LinearCache(activations, weights, bias))
  }

  def linearForwardActivation(activationsPrev: DenseMatrix[Double], 
                              weights: DenseMatrix[Double], 
                              bias: DenseVector[Double], 
                              activation: ActivationFunction)
  : (DenseMatrix[Double], (LinearCache, ActivationCache)) = {
    ???
  }

  def lModelForward(XInput: DenseMatrix[Double], parameters: Parameters): (DenseMatrix[Double], Cache) = ???
}


