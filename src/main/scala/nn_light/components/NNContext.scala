package nn_light.components

trait NNContext {
  def layersDims: Seq[Int]
  def learningRate: Double 
  def numIterations: Int
  def initializer: Initializer
  def forwardActivation: ForwardActivation
  def costFunction: CostFunction
  def backwardActivation: BackwardActivation
  def optimizer: Optimizer
}


case class SimpleNNContext(layersDims: Seq[Int], 
                           learningRate: Double,
                           numIterations: Int,
                           initializer: Initializer,
                           forwardActivation: ForwardActivation,
                           costFunction: CostFunction,
                           backwardActivation: BackwardActivation,
                           optimizer: Optimizer
                          ) extends NNContext
