package nn_light.components

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sqrt}

trait Optimizer {
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double], 
               Yinput: DenseMatrix[Double],
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Grads, Double)): (Parameters, Seq[Double])
}

class GradientDescentOptimizer(numIterations: Int, learningRate: Double, costLimit: Double = 0.06) 
extends 
  Optimizer {
  
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double],
               Yinput: DenseMatrix[Double],
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Grads, Double)): (Parameters, Seq[Double]) = {
    var parameters = initialParameters
    var cost = Double.MaxValue
    var costs = Seq[Double]()
    for {
      iteration <- 0 until numIterations
      if cost > costLimit
    } {
      val (grads, newCost) = provider(parameters, Xinput, Yinput)
      if (iteration % 1000 == 0) {
        println(s"cost at iteration: $iteration -> $newCost")
        costs = costs :+ newCost
      }
      cost = newCost
      parameters = parameters.update(grads, learningRate)
    }
    (parameters, costs)
  }
}

class RandomMiniBatchGradientDescentOptimizer(batchSize: Int, 
                                              numOfEpochs: Int, 
                                              learningRate: Double,
                                              costLimit: Double = 0.06) extends Optimizer {
  
  private def shuffleData(Xinput: DenseMatrix[Double], 
                          Yinput: DenseMatrix[Double])
  : Seq[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val m = Xinput.cols
    val permutations = MathUtils.randomPermutation(m)

    val XShuffled = Xinput(::, permutations).toDenseMatrix
    val YShuffled = Yinput(::, permutations).toDenseMatrix

    val completeBatches = m / batchSize

    var batches = (0 until completeBatches)
      .foldLeft(Seq[(DenseMatrix[Double],DenseMatrix[Double])]()) { (seq, offset) =>
        val batchIndices = offset * batchSize until  (offset + 1) * batchSize
        val Xbatch = XShuffled(::, batchIndices).toDenseMatrix
        val Ybatch = YShuffled(::, batchIndices).toDenseMatrix
        seq :+ (Xbatch, Ybatch)
      }

    if (m % batchSize != 0) {
      val Xbatch = XShuffled(::, completeBatches * batchSize  until m ).toDenseMatrix
      val Ybatch = YShuffled(::, completeBatches * batchSize  until m ).toDenseMatrix
      batches = batches :+ (Xbatch, Ybatch)
    }
    batches
  }
  
  protected def updateParameters (parameters: Parameters, grads: Grads, 
                                  adamCounter: Int): Parameters = {
    parameters.update(grads, learningRate)
  }
  
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double], 
               Yinput: DenseMatrix[Double], 
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Grads, Double)): (Parameters, Seq[Double]) = {
    val m = Xinput.cols
    val completeBatches = m / batchSize
    
    val batches = shuffleData(Xinput, Yinput)
    var parameters = initialParameters
    var costs = Seq[Double]()
    var cost = Double.MaxValue
    var adamCounter = 0
    for {
      epoch <- 0 until numOfEpochs
      ((xBatch, yBatch), idx) <- batches.zipWithIndex
      if cost > costLimit
    } {
      val (grads, newCost) = provider(parameters, xBatch, yBatch)
      
      if (epoch % 1000 == 0 && idx == completeBatches - 1) {
        println(s"cost at epoch: $epoch -> $newCost")
        costs = costs :+ newCost
      }
      cost = newCost
      adamCounter += 1
      parameters = updateParameters(parameters, grads, adamCounter)
    }
    (parameters, costs)
  }
}


class MomentumRandomMiniBatchGradientDescentOptimizer (batchSize: Int,
                                                       numOfEpochs: Int,
                                                       learningRate: Double,
                                                       beta:Double,
                                                       costLimit: Double = 0.06) extends 
RandomMiniBatchGradientDescentOptimizer(batchSize, numOfEpochs, learningRate, costLimit) {
 
  
  private var vW = Map[String, DenseMatrix[Double]]()
  private var vb = Map[String, DenseVector[Double]]()

  override protected def updateParameters(parameters: Parameters, grads: Grads, 
                                          adamCounter: Int): Parameters = {
    parameters.weights.keys.foreach { key =>
      val gradKey = s"d$key"
      val updatedVW = (beta * vW(gradKey)) + ((1 - beta) * grads.matrices(gradKey))
      vW = vW.updated(gradKey, updatedVW)
    }

    parameters.bias.keys.foreach { key =>
      val gradKey = s"d$key"
      val updatedVW = (beta * vb(gradKey)) + ((1 - beta) * grads.vectors(gradKey))
      vb = vb.updated(gradKey, updatedVW)
    }
    
    val momentumGrads = Grads(vW, vb)
    parameters.update(momentumGrads, learningRate)
  }

  override def optimize(initialParameters: Parameters, 
                        Xinput: DenseMatrix[Double], 
                        Yinput: DenseMatrix[Double], 
                        provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double])
                          => (Grads, Double)): (Parameters, Seq[Double]) = {
    
    vW = initialParameters.weights.keys.foldLeft(vW) { (v, key) => 
      val w = initialParameters.weights(key)
      v + (s"d$key" -> DenseMatrix.zeros[Double](w.rows, w.cols))
    }
    vb = initialParameters.bias.keys.foldLeft(vb) { (v, key) =>
      val b = initialParameters.bias(key)
      v + (s"d$key" -> DenseVector.zeros[Double](b.length))
    }
    
    super.optimize(initialParameters, Xinput, Yinput, provider)
  }
}

class RandomMiniBatchWithADAMOptimizer (batchSize: Int,
                                        numOfEpochs: Int,
                                        learningRate: Double,
                                        beta1: Double,
                                        beta2: Double,
                                        epsilon: Double,
                                        costLimit: Double = 0.06) extends 
  RandomMiniBatchGradientDescentOptimizer (batchSize, numOfEpochs, learningRate, costLimit) {
  private var vW = Map[String, DenseMatrix[Double]]()
  private var vb = Map[String, DenseVector[Double]]()
  private var sW = Map[String, DenseMatrix[Double]]()
  private var sb = Map[String, DenseVector[Double]]()

  override protected def updateParameters(parameters: Parameters, grads: Grads, 
                                          adamCounter: Int): Parameters = {
    var gradW = Map[String, DenseMatrix[Double]]()
    var gradB = Map[String, DenseVector[Double]]()

    parameters.weights.keys.foreach { key =>
      val gradKey = s"d$key"
      val grad = grads.matrices(gradKey)
      val updatedVW = (beta1 * vW(gradKey)) + ((1 - beta1) * grad)
      val updatedSW = (beta2 * sW(gradKey)) + ((1 - beta2) * pow(grad, 2))
      val correctedVW = updatedVW / (1 - pow(beta1, adamCounter))
      val correctedSW = updatedSW / (1 - pow(beta2, adamCounter))
      vW = vW.updated(gradKey, updatedVW)
      sW = sW.updated(gradKey, updatedSW)
      gradW = gradW.updated(gradKey, correctedVW / sqrt(correctedSW + epsilon))
    }

    parameters.bias.keys.foreach { key =>
      val gradKey = s"d$key"
      val grad = grads.vectors(gradKey)
      val updatedVb = (beta1 * vb(gradKey)) + ((1 - beta1) * grad)
      val updatedSb = (beta2 * sb(gradKey)) + ((1 - beta2) * pow(grad, 2))
      val correctedVb = updatedVb / (1 - pow(beta1, adamCounter))
      val correctedSb = updatedSb / (1 - pow(beta2, adamCounter))
      vb = vb.updated(gradKey, updatedVb)
      sb = sb.updated(gradKey, updatedSb)
      gradB = gradB.updated(gradKey, correctedVb / (sqrt(correctedSb) + epsilon))
    }
    
    val newGrads = Grads(gradW, gradB)
    parameters.update(newGrads, learningRate)
  }

  override def optimize(initialParameters: Parameters, 
                        Xinput: DenseMatrix[Double], 
                        Yinput: DenseMatrix[Double], 
                        provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                          (Grads, Double)): (Parameters, Seq[Double]) = {

    val (initVW, initSW) = initialParameters.weights.keys.foldLeft((vW, sW)) { (v, key) =>
      val w = initialParameters.weights(key)
      (v._1 + (s"d$key" -> DenseMatrix.zeros[Double](w.rows, w.cols)), 
       v._2 + (s"d$key" -> DenseMatrix.zeros[Double](w.rows, w.cols)))
    }
    vW = initVW
    sW = initSW
    
    val (initVb, initSb) = initialParameters.bias.keys.foldLeft((vb, sb)) { (v, key) =>
      val b = initialParameters.bias(key)
      (v._1 + (s"d$key" -> DenseVector.zeros[Double](b.length)), 
       v._2 + (s"d$key" -> DenseVector.zeros[Double](b.length)))
    }
    vb = initVb
    sb = initSb
    super.optimize(initialParameters, Xinput, Yinput, provider)
  }
}
