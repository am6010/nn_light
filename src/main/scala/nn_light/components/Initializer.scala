package nn_light.components

import breeze.linalg.{DenseMatrix, DenseVector}

trait Initializer {
  def initializeParametersDeep(layersDims: Seq[Int]): Parameters 
}

case class RandomInitializer() extends Initializer {
  def initializeParametersDeep(layersDims: Seq[Int]): Parameters = {
    if (layersDims.isEmpty || layersDims.lengthCompare(1) == 0) {
      throw new RuntimeException("Layers should be greater than one")
    }
    
    val weightsInit = Map[String, DenseMatrix[Double]] ()
    val biasInit = Map[String, DenseVector[Double]] ()
    
    val zippedLayersDims = layersDims.zip(layersDims.tail).zipWithIndex
    
    val weights = zippedLayersDims.foldLeft(weightsInit) { case (w, ((li_1, li), idx)) =>
      val lWeight: DenseMatrix[Double] = DenseMatrix.rand(li, li_1) *:* 0.01
      w + (s"W${idx + 1}" -> lWeight)
    }
    
    val bias = layersDims.tail.zipWithIndex.foldLeft(biasInit) { case (b, (li, idx)) =>
      val lBias: DenseVector[Double] = DenseVector.zeros(li)
      b + (s"b${idx + 1}" -> lBias)
    }
    
    Parameters(weights, bias)
  }
}

case class Parameters(weights: Map[String, DenseMatrix[Double]],
                      bias: Map[String, DenseVector[Double]])
