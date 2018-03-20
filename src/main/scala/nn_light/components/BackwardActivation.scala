package nn_light.components

import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}

case class Derivatives(dAPrev: DenseMatrix[Double], 
                       dW: DenseMatrix[Double], 
                       db: DenseVector[Double])

trait BackwardActivation {
  
  def linearBackward(dZ: DenseMatrix[Double], linearCache: LinearCache): Derivatives
  
  def linearActivationBackward(dA: DenseMatrix[Double], 
                               cache:(LinearCache, ActivationCache), 
                               activationFunction: ActivationFunction): Derivatives
  
  def lModelBackward(aL: DenseMatrix[Double], 
                     y: DenseMatrix[Double], 
                     cache: Cache): Map[String, DenseMatrix[Double]] 
}


class BackwardActivationImpl extends BackwardActivation {
  
  def linearBackward(dZ: DenseMatrix[Double], linearCache: LinearCache): Derivatives = {
    val aPrev = linearCache.activations
    val weights = linearCache.weights
    val m = aPrev.cols.toDouble
    val dW = dZ * aPrev.t
    val normDW = dW /:/ DenseMatrix.fill(dW.rows, dW.cols){m}
    val db = sum(dZ, Axis._1) 
    val normDb  = db /:/ DenseVector.fill(db.length){m}
    val dAPrev = weights.t * dZ
    Derivatives(dAPrev, normDW, normDb)
  }

  def linearActivationBackward(dA: DenseMatrix[Double], 
                               cache: (LinearCache, ActivationCache), 
                               activationFunction: ActivationFunction): Derivatives = {
    ???
  }

  def lModelBackward(aL: DenseMatrix[Double], 
                     y: DenseMatrix[Double], 
                     cache: Cache): Map[String, DenseMatrix[Double]] = {
    ???
  }
}


object BackwardActivationImpl {
  def apply(): BackwardActivationImpl = new BackwardActivationImpl()
}
