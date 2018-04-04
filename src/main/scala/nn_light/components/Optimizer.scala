package nn_light.components

import breeze.linalg.DenseMatrix

trait Optimizer {
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double], 
               Yinput: DenseMatrix[Double],
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Parameters, Double)): (Parameters, Seq[Double])
}

class GradientDescentOptimizer(numIterations: Int, costLimit: Double = 0.06) extends Optimizer {
  
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double],
               Yinput: DenseMatrix[Double],
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Parameters, Double)): (Parameters, Seq[Double]) = {
    var parameters = initialParameters
    var cost = Double.MaxValue
    var costs = Seq[Double]()
    for {
      iteration <- 0 until numIterations
      if cost > costLimit
    } {
      val (newParameters, newCost) = provider(parameters, Xinput, Yinput)
      if (iteration % 1000 == 0) {
        println(s"cost at iteration: $iteration -> $newCost")
        costs = costs :+ newCost
      }
      cost = newCost
      parameters = newParameters
    }
    (parameters, costs)
  }
}

class RandomMiniBatchGradientDescentOptimizer(batchSize: Int, numOfEpochs: Int, 
                                              costLimit: Double = 0.06) extends Optimizer {
  
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double], 
               Yinput: DenseMatrix[Double], 
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Parameters, Double)): (Parameters, Seq[Double]) = {
    
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
    
    var parameters = initialParameters
    var costs = Seq[Double]()
    var cost = Double.MaxValue
    for {
      epoch <- 0 until numOfEpochs
      ((xBatch, yBatch), idx) <- batches.zipWithIndex
      if cost > costLimit
    } {
      val (newParameters, newCost) = provider(parameters, xBatch, yBatch)
      
      if (epoch % 1000 == 0 && idx == completeBatches - 1) {
        println(s"cost at epoch: $epoch -> $newCost")
        costs = costs :+ newCost
      }
      cost = newCost
      parameters = newParameters
    }
    (parameters, costs)
  }
}
