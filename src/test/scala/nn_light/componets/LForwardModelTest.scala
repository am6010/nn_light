package nn_light.componets

import breeze.linalg.{DenseMatrix, DenseVector}
import nn_light.components._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class LForwardModelTest extends FunSuite {

  trait setUp {
    val forwardModel = new LForwardModel()
    val delta =  0.000001
  }
  
  test("linearForward test one vector example") {
    new setUp {
      val aPrev = DenseMatrix(0.3 , -1.3)
      val weights = DenseMatrix((0.2, -1.4), (-0.12, 0.9))
      val bias = DenseVector(1.0, 0.5)
      val (z, linearCache) = forwardModel.linearForward(aPrev, weights, bias)
      assert(z !== null)
      assert(linearCache === LinearCache(aPrev, weights, bias))
      assert(z.rows === 2)
      assert(math.abs(z(0, 0) - ((0.3 * 0.2 + 1.3 * 1.4) + 1)) < delta)
      assert(math.abs(z(1, 0) - ((-0.3 * 0.12 - 1.3 * 0.9) + 0.5)) < delta)
    }
  }

  test("linearForward test 2 examples") {
    new setUp {
      val aPrev = DenseMatrix((0.3 , -1.3), (0.4, 1.0))
      val weights = DenseMatrix((0.2, -1.4), (-0.12, 0.9))
      val bias = DenseVector(1.0, 0.5)
      val (z, linearCache) = forwardModel.linearForward(aPrev, weights, bias)
      assert(z !== null)
      assert(linearCache === LinearCache(aPrev, weights, bias))
      assert(z.rows === 2 && z.cols === 2)
      assert(math.abs(z(0, 0) - ((0.3 * 0.2 - 0.4 * 1.4) + 1)) < delta)
      assert(math.abs(z(1, 1) - ((1.3 * 0.12 + 1.0 * 0.9) + 0.5)) < delta)
    }
  }
  
  test("linearForwardActivation test ") {
    new setUp {
      val aPrev = DenseMatrix((0.3 , -1.3), (0.4, 1.0))
      val weights = DenseMatrix((0.2, -1.4), (-0.12, 0.9))
      val bias = DenseVector(1.0, 0.5)
      val activationFun = Relu()
      val (aL, caches) = forwardModel.linearForwardActivation(aPrev, weights, bias, activationFun)
      val (linearCache, activationCache) = caches
      assert(linearCache === LinearCache(aPrev, weights, bias))
      assert(aL.rows === 2 && aL.cols === 2)
      val (zInput, _) = forwardModel.linearForward(aPrev, weights, bias)
      assert(zInput === activationCache.inputs)
    }
  }
  
  test("LForwardModel test") {
    new setUp {
      val XInput = DenseMatrix((0.3 , -1.3), (0.4, 1.0))
      val parameters: Parameters = RandomInitializer().initializeParametersDeep(Seq(2, 4, 5, 1))
      val (yOut, caches) = forwardModel.lModelForward(XInput = XInput, parameters = parameters)
      assert(yOut.rows === 1 && yOut.cols === 2)
      
      
    }
  }
}
