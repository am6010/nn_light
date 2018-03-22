package nn_light.components

import breeze.linalg.DenseMatrix


trait NNModel {
  
  def train(X: DenseMatrix[Double], Y: DenseMatrix[Double]): Seq[Double]
  
  def predict(X: DenseMatrix[Double]): DenseMatrix[Double]
}

class DeepNN(context: NNContext) extends NNModel {
  
  private val layersDims = context.layersDims
  private val optimizer = context.optimizer
  private var parameters: Parameters = _
  
  private val forwardActivation = context.forwardActivation
  private val costFunction = context.costFunction
  private val backwardActivation = context.backwardActivation
  private val learningRate = context.learningRate
  
  def train(X: DenseMatrix[Double], Y: DenseMatrix[Double]): Seq[Double] = {
    val startParams = context.initializer.initializeParametersDeep(layersDims)
    val (trainedParams, costs) = optimizer.optimize(startParams, params => {
      val(aL, cache) =  forwardActivation.lModelForward(X, params)
      val cost = costFunction.computeCost(aL, Y, params)
      val grads = backwardActivation.lModelBackward(aL, Y, cache)
      (params.update(grads, learningRate), cost)
    })
    parameters = trainedParams
    costs
  }

  def predict(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val (predictions, _) = forwardActivation.lModelForward(X, parameters)
    predictions.map(x => if (x > 0.5) 1.0 else 0.0)
  }
}

object DeepNN {
  def apply(context: NNContext): DeepNN = new DeepNN(context)
}
