package nn_light.components

import breeze.linalg.{DenseMatrix, DenseVector}

case class LinearCache(activations: DenseMatrix[Double], 
                       weights: DenseMatrix[Double],
                       bias: DenseVector[Double])

case class ActivationCache(activations: DenseMatrix[Double])

case class Cache(caches: Seq[(LinearCache, ActivationCache)]) {
  def add(cache: (LinearCache, ActivationCache)): Cache = {
    Cache(caches :+ cache)
  }
}
