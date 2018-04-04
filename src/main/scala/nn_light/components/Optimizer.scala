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

class RandomMiniBatchGradientDescentOptimizer(batchSize: Int) extends Optimizer {
  
  def optimize(initialParameters: Parameters, 
               Xinput: DenseMatrix[Double], 
               Yinput: DenseMatrix[Double], 
               provider: (Parameters, DenseMatrix[Double], DenseMatrix[Double]) => 
                 (Parameters, Double)): (Parameters, Seq[Double]) = {
    ???  
  }
}
