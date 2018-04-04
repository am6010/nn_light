package nn_light.components

import breeze.linalg.DenseMatrix

import scala.util.Random

object MathUtils {
  def removeInfiniteAndNans(input: DenseMatrix[Double]) : DenseMatrix[Double] = {
    input.mapValues {
      case x if x.isPosInfinity => 1e8
      case x if x.isNegInfinity => -1e8
      case x if x.isNaN => 0.0
      case x => x
    }
  }
  
  def randomPermutation(limit: Int): Seq[Int] =  {
    val numbers = (0 until limit).toArray
    val rd = new Random 
    for {
      i <- numbers.indices
      n = rd.nextInt(i + 1)
    }  {
      numbers(i) = numbers(n)
      numbers(n) = i
    }
    numbers
  }
}
