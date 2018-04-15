package nn_light.components

trait NNContext {
  def layersDims: Seq[Int]
  def numIterations: Int
  def initializer: Initializer
  def forwardActivation: ForwardActivation
  def costFunction: CostFunction
  def backwardActivation: BackwardActivation
  def optimizer: Optimizer
}


case class SimpleNNContext(layersDims: Seq[Int],
                           numIterations: Int,
                           initializer: Initializer,
                           forwardActivation: ForwardActivation,
                           costFunction: CostFunction,
                           backwardActivation: BackwardActivation,
                           optimizer: Optimizer
                          ) extends NNContext
