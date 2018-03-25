package nn_light.components

import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}

case class Derivatives(dAPrev: DenseMatrix[Double], 
                       dW: DenseMatrix[Double], 
                       db: DenseVector[Double])

case class Grads(matrices: Map[String, DenseMatrix[Double]], 
                 vectors: Map[String, DenseVector[Double]]) {
  
  def update(devs: Derivatives, daLIdx: Int, layer:Int) : Grads = {
    val newMatrices = matrices + (s"dA$daLIdx" -> devs.dAPrev)  + (s"dW$layer" -> devs.dW)
    val newVectors = vectors + (s"db$layer" -> devs.db)
    Grads(newMatrices, newVectors)
  }
}

trait BackwardActivation {
  
  def linearBackward(dZ: DenseMatrix[Double], linearCache: LinearCache): Derivatives
  
  def linearActivationBackward(dA: DenseMatrix[Double], 
                               cache:(LinearCache, ActivationCache), 
                               activationFunction: ActivationFunction): Derivatives
  
  def lModelBackward(aL: DenseMatrix[Double], 
                     y: DenseMatrix[Double], 
                     cache: Cache): Grads
}


class BackwardActivationImpl extends BackwardActivation {
  
  def linearBackward(dZ: DenseMatrix[Double], linearCache: LinearCache): Derivatives = {
    val aPrev = linearCache.activations
    val weights = linearCache.weights
    val m = aPrev.cols.toDouble
    val dW = (dZ * aPrev.t) /:/ m
    val db = sum(dZ, Axis._1)  /:/ m
    val dAPrev = weights.t * dZ
    Derivatives(dAPrev, dW, db)
  }

  def linearActivationBackward(dA: DenseMatrix[Double], 
                               cache: (LinearCache, ActivationCache), 
                               activationFunction: ActivationFunction): Derivatives = {
    val (linearCache, activationCache) = cache
    val dZ = dA *:* activationFunction.firstDerivative(activationCache.inputs)
    this.linearBackward(dZ, linearCache)
  }

  def lModelBackward(aL: DenseMatrix[Double], 
                     y: DenseMatrix[Double], 
                     cache: Cache): Grads = {
    val L = cache.caches.size
    val dAL = - ((y /:/ aL) - ((1.0 - y) /:/ (1.0 - aL))) 
    val lastCache = cache.caches(L - 1)
    val derivatives = linearActivationBackward(dAL, lastCache, Sigmoid())
    val lastLayerGrads = Grads(Map(), Map()).update(derivatives, L-1, L)
    (L-2 to 0 by -1).foldLeft(lastLayerGrads) { (grads, l) =>  
      val currentCache = cache.caches(l)
      val currentGrads = linearActivationBackward(grads.matrices(s"dA${l+1}"), currentCache, Relu())
      grads.update(currentGrads, l, l+1)
    }
  }
}

object BackwardActivationImpl {
  def apply(): BackwardActivationImpl = new BackwardActivationImpl()
}


class BackwardActivationL2Impl(lambda: Double) extends BackwardActivation {

  def linearBackward(dZ: DenseMatrix[Double], linearCache: LinearCache): Derivatives = {
    val aPrev = linearCache.activations
    val weights = linearCache.weights
    val m = aPrev.cols.toDouble
    val dW = dZ * aPrev.t
    val normDW = dW *:* (1.0 / m)  + ((lambda/m) * weights)  
    val db = sum(dZ, Axis._1)
    val normDb  = db *:* (1.0 / m)
    val dAPrev = weights.t * dZ
    Derivatives(dAPrev, normDW, normDb)
  }

  def linearActivationBackward(dA: DenseMatrix[Double],
                               cache: (LinearCache, ActivationCache),
                               activationFunction: ActivationFunction): Derivatives = {
    val (linearCache, activationCache) = cache
    val dZ = dA *:* activationFunction.firstDerivative(activationCache.inputs)
    this.linearBackward(dZ, linearCache)
  }

  def lModelBackward(aL: DenseMatrix[Double],
                     y: DenseMatrix[Double],
                     cache: Cache): Grads = {
    val L = cache.caches.size
    val ones = DenseMatrix.ones[Double](y.rows, y.cols)
    val dAL = - ((y /:/ aL) - ((ones - y) /:/ (ones - aL)))
    val lastCache = cache.caches(L - 1)
    val derivatives = linearActivationBackward(dAL, lastCache, Sigmoid())
    val lastLayerGrads = Grads(Map(), Map()).update(derivatives, L-1, L)
    (L-2 to 0 by -1).foldLeft(lastLayerGrads) { (grads, l) =>
      val currentCache = cache.caches(l)
      val currentGrads = linearActivationBackward(grads.matrices(s"dA${l+1}"), currentCache, Relu())
      grads.update(currentGrads, l, l+1)
    }
  }
}
