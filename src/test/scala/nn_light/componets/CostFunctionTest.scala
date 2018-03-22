package nn_light.componets

import breeze.linalg.DenseMatrix
import nn_light.components.EntropyCostFunction
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class CostFunctionTest extends FunSuite {
  
  trait SetUp {
    val costFunction = new EntropyCostFunction()
  }
  
  test("Empty inputs for const function") {
   new SetUp {
     assertThrows[RuntimeException] {
       val cost = costFunction.computeCost(DenseMatrix.ones(0, 0), DenseMatrix.ones(0, 0), null)
       println(cost)
     }
   } 
  }
  
  test("Entropy cost function test") {
    new SetUp {
      val aL = DenseMatrix((0.98, 0.01, 0.002))
      val y = DenseMatrix((1.0, 0.0, 0.0))
      val cost: Double = costFunction.computeCost(aL, y, null)
      val expected: Double = math.log(0.98) + math.log(0.99) + math.log(0.998)
      assert(cost === - (expected / 3.0))
    }
  }

  test("Entropy cost function test2") {
    new SetUp {
      val aL = DenseMatrix((0.8, 0.9, 0.4))
      val y = DenseMatrix((1.0, 1.0, 1.0))
      val cost: Double = costFunction.computeCost(aL, y, null)
      val expected: Double = math.log(0.98) + math.log(0.99) + math.log(0.998)
      assert(math.abs(cost - 0.414931599615) < 0.0000001)
    }
  }
}
