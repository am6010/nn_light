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
    val (zInput, linearCache) = linearForward(activationsPrev, weights, bias)
    val aL = activation.activate(zInput)
    (aL, (linearCache, ActivationCache(zInput)))
  }

  def lModelForward(XInput: DenseMatrix[Double], 
                    parameters: Parameters): (DenseMatrix[Double], Cache) = {
    
    val layers = parameters.weights.size
    val hiddenLayers = (1 until layers)
      .foldLeft((XInput, Cache(Seq()))) { case ((aLPrev, caches ), idx) =>
      val weights = parameters.weights(s"W$idx")
      val bias = parameters.bias(s"b$idx")
      val (aL, cache) = linearForwardActivation(aLPrev, weights, bias, Relu())
      (aL, caches.add(cache))
    }

    val weights = parameters.weights(s"W$layers")
    val bias = parameters.bias(s"b$layers")
    val (yHat, cache) = linearForwardActivation(hiddenLayers._1, weights, bias, Sigmoid())
    (yHat, hiddenLayers._2.add(cache))
  }
}


