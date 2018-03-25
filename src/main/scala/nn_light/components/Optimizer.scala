package nn_light.components

trait Optimizer {
  def optimize(initialParameters: Parameters, 
               provider: Parameters => (Parameters, Double)): (Parameters, Seq[Double])
}

class GradientDescentOptimizer(numIterations: Int) extends Optimizer {
  def optimize(initialParameters: Parameters, 
               provider: Parameters => (Parameters, Double)): (Parameters, Seq[Double]) = {
    
    (0 until numIterations).foldLeft((initialParameters, Seq[Double]())) 
    { case ((params, costs), iter) =>
      val (newParams, newCost) = provider(params)
      
      if (iter % 1000 == 0) {
        println(s"cost at iteration: $iter -> $newCost")
      }
      (newParams, costs :+ newCost)
    }
  }
}
