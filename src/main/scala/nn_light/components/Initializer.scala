package nn_light.components

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian, Rand}

trait Initializer {
  def initializeParametersDeep(layersDims: Seq[Int]): Parameters
  
  protected def checkInput(layersDims: Seq[Int]): Unit = {
    if (layersDims.isEmpty || layersDims.lengthCompare(1) == 0) {
      throw new RuntimeException("Layers should be greater than one")
    } 
  }
  
  protected def initialiseParams(layersDims: Seq[Int], multipliers: Seq[Double]): Parameters = {
    val weightsInit = Map[String, DenseMatrix[Double]] ()
    val biasInit = Map[String, DenseVector[Double]] ()

    val zippedLayersDims = layersDims.zip(layersDims.tail).zipWithIndex

    val weights = zippedLayersDims.foldLeft(weightsInit) { case (w, ((li_1, li), idx)) =>
      val normal01 = Gaussian(0,1)
      val lWeight: DenseMatrix[Double] = 
        DenseMatrix.rand(li, li_1, normal01) *:* multipliers(idx)
      w + (s"W${idx + 1}" -> lWeight)
    }

    val bias = layersDims.tail.zipWithIndex.foldLeft(biasInit) { case (b, (li, idx)) =>
      val lBias: DenseVector[Double] = DenseVector.zeros(li)
      b + (s"b${idx + 1}" -> lBias)
    }

    Parameters(weights, bias)
  }
}

case class RandomInitializer(multiplier: Double) extends Initializer {
  def initializeParametersDeep(layersDims: Seq[Int]): Parameters = {
    checkInput(layersDims)
    val multipliers = layersDims.tail.map(_ => multiplier)
    initialiseParams(layersDims, multipliers)
  }
}

case class HeInitializer() extends Initializer {
  def initializeParametersDeep(layersDims: Seq[Int]): Parameters = {
    checkInput(layersDims)
    val multipliers = layersDims.take(layersDims.size - 1).map(1.0 / sqrt(_))
    initialiseParams(layersDims, multipliers)
  }
}

case class Parameters(weights: Map[String, DenseMatrix[Double]],
                      bias: Map[String, DenseVector[Double]]) {
  
  def update(grads: Grads, learningRate: Double): Parameters = {
    val newWeights = weights.map { case (key, w) =>
      key -> (w - (learningRate * grads.matrices(s"d$key")))
    }
    
    val newBias = bias.map { case (key, b) =>
      key -> (b - (learningRate * grads.vectors(s"d$key")))
    }
    
    Parameters(newWeights, newBias)
  }
}
