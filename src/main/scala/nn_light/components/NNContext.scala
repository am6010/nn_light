package nn_light.components

trait NNContext {
  def layersDims: Seq[Int]
  def learningRate: Double 
  def numIterations: Int
}


case class SimpleNNContext(layersDims: Seq[Int], 
                           learningRate: Double,
                           numIterations: Int) extends NNContext
