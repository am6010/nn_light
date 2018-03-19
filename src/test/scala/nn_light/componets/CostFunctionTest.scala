package nn_light.componets

import breeze.linalg.DenseMatrix
import nn_light.components.EntropyCostFunction
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class CostFunctionTest extends FunSuite {
  
  test("Entropy cost function test") {
    val costFunction = new EntropyCostFunction()
    
    val aL = DenseMatrix((0.98, 0.01, 0.002))
    val y = DenseMatrix((1.0, 0.0, 0.0))
    val cost = costFunction.computeCost(aL, y)
    val expected = math.log(0.98) + math.log(0.99) + math.log(0.998)
    assert(cost === expected / 3.0)
  }

}
