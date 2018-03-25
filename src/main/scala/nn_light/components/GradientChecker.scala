package nn_light.components

import breeze.linalg.{DenseMatrix, DenseVector, norm}

object GradientChecker {

  def vectorizeParameters(params: Parameters, L: Int): DenseVector[Double] = {
    (1 to L).foldLeft(DenseVector[Double]()) {(vec, l) =>
      val w = params.weights(s"W$l").t.toDenseVector
      val b = params.bias(s"b$l")
      val all = DenseVector.vertcat(w, b)
      DenseVector.vertcat(vec, all)
    }
  }

  def vectorizeGrads(grads: Grads, L: Int): DenseVector[Double] = {
    (1 to L).foldLeft(DenseVector[Double]()) {(vec, l) =>
      val w = grads.matrices(s"dW$l").t.toDenseVector
      val b = grads.vectors(s"db$l")
      val all = DenseVector.vertcat(w, b)
      DenseVector.vertcat(vec, all)
    }
  }
  
  def vectorToParameters(vector: DenseVector[Double], layers: Seq[Int]): Parameters = {
    val initW = Map[String, DenseMatrix[Double]]()
    val initB = Map[String, DenseVector[Double]]()
    var i = 0
    val maps = layers.zip(layers.tail).foldLeft((initW, initB, 0)) { 
      case ((ws, bs, base), (lc, ln)) => 
        val prod = lc * ln
        val w = vector(base until base + prod).asDenseMatrix.reshape(lc, ln).t
        val b = vector(base + prod until base + prod + ln)
        i += 1
        (ws + (s"W$i" -> w), bs + (s"b$i" -> b), base + prod + ln)
    }
    Parameters(maps._1, maps._2)
  }
  
  def gradientChecking(X: DenseMatrix[Double], 
                       Y: DenseMatrix[Double],
                       layers: Seq[Int],
                       parameters: Parameters,
                       forward: ForwardActivation,
                       costFunction: CostFunction,
                       backward: BackwardActivation
                      ): Double = {
    val paramVec = vectorizeParameters(parameters, layers.size - 1)
    val (aL, cache) = forward.lModelForward(X, parameters)
    val cost = costFunction.computeCost(aL, Y, parameters)
    val grads = backward.lModelBackward(aL, Y, cache)
    val vecGrads = vectorizeGrads(grads, layers.size - 1)
    val epsilon = 1e-7

    val gradappxArr = (0 until paramVec.length).map { ith =>
      val thetaPlus = paramVec.copy
      thetaPlus(ith) = thetaPlus(ith) + epsilon
      val paramsPlus = vectorToParameters(thetaPlus, layers)
      val (aLPlus, _) = forward.lModelForward(X, paramsPlus)
      val costPlus = costFunction.computeCost(aLPlus, Y, paramsPlus)

      val thetaMinus = paramVec.copy
      thetaMinus(ith) = thetaMinus(ith) - epsilon
      val paramMinus = vectorToParameters(thetaMinus, layers)
      val (aLMinus, _) = forward.lModelForward(X, paramMinus)
      val costMinus = costFunction.computeCost(aLMinus, Y, paramMinus)

      (costPlus - costMinus) / (2 * epsilon)
    }.toArray

    val gradappx = DenseVector(gradappxArr)
    val numerator = norm(vecGrads - gradappx)
    val denominator = norm(vecGrads) + norm(gradappx)

    numerator / denominator
  }
}

