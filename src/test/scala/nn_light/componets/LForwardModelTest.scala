package nn_light.componets

import breeze.linalg.{DenseMatrix, DenseVector}
import nn_light.components._
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FunSuite}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class LForwardModelTest extends FunSuite {

  trait setUp {
    val forwardModel = new LForwardModel()
    val delta =  0.000001

    def testMatrixValues(actual: DenseMatrix[Double], expected: DenseMatrix[Double]): Assertion = {
      val  dif = actual -:- expected <:< DenseMatrix.fill(actual.rows, actual.cols){0.0000001}
      assert(dif.forall(x => x))
    }
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

  test("linearForward test 3") {
    new setUp {
      val aPrev = DenseMatrix((1.62434536, -0.61175641),
        (-0.52817175, -1.07296862),
        (0.86540763, -2.3015387))
      val weights = DenseMatrix((1.74481176, -0.7612069,   0.3190391))
      val bias = DenseVector(-0.24937038)
      val (z, linearCache) = forwardModel.linearForward(aPrev, weights, bias)
      testMatrixValues(z, DenseMatrix((3.26295337, -1.23429987)))
      testMatrixValues(linearCache.activations, aPrev)
      testMatrixValues(linearCache.weights, weights)
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


  test("linearForwardActivation test relu") {
    new setUp {
      val aPrev = DenseMatrix((-0.41675785, -0.05626683), 
        (-2.1361961, 1.64027081), 
        (-1.79343559, -0.84174737))
      val weights = DenseMatrix((0.50288142, -1.24528809, -1.05795222))
      val bias = DenseVector(-0.90900761)
      val activationFun = Relu()
      val (aL, caches) = forwardModel.linearForwardActivation(aPrev, weights, bias, activationFun)
      val (linearCache, activationCache) = caches
      assert(linearCache === LinearCache(aPrev, weights, bias))
      testMatrixValues(aL, DenseMatrix((3.43896131, 0.0)))
    }
  }


  test("linearForwardActivation test sigmoid") {
    new setUp {
      val aPrev = DenseMatrix((-0.41675785, -0.05626683),
        (-2.1361961, 1.64027081),
        (-1.79343559, -0.84174737))
      val weights = DenseMatrix((0.50288142, -1.24528809, -1.05795222))
      val bias = DenseVector(-0.90900761)
      val activationFun = Sigmoid()
      val (aL, caches) = forwardModel.linearForwardActivation(aPrev, weights, bias, activationFun)
      val (linearCache, activationCache) = caches
      assert(linearCache === LinearCache(aPrev, weights, bias))
      testMatrixValues(aL, DenseMatrix((0.96890023,  0.11013289)))
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

  test("LForwardModel test 2") {
    new setUp {
      val XInput = DenseMatrix((-0.31178367,  0.72900392,  0.21782079, -0.8990918),
      (-2.48678065,  0.91325152, 1.12706373, -1.51409323),
      ( 1.63929108, -0.4298936,  2.63128056,  0.60182225),
      (-0.33588161,  1.23773784,  0.11112817,  0.12915125),
      (0.07612761, -0.15512816,  0.63422534,  0.810655))
      val W1 = DenseMatrix((0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384),
        (-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953),
        (-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143),
        (-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059))
      val W2 = DenseMatrix((-0.12673638, -1.36861282,  1.21848065, -0.85750144),
        (-0.56147088, -1.0335199 ,  0.35877096,  1.07368134),
        (-0.37550472,  0.39636757, -0.47144628,  2.33660781))
      val W3 = DenseMatrix((0.9398248 ,  0.42628539, -0.75815703))
      
      val b1 = DenseVector(1.38503523, -0.51962709, -0.78015214, 0.95560959)
      val b2 = DenseVector(1.50278553, -0.59545972, 0.52834106)
      val b3 = DenseVector(-0.16236698)
      val parameters: Parameters = Parameters(Map(("W1", W1), ("W2", W2), ("W3", W3)),
        Map(("b1", b1), ("b2", b2), ("b3", b3)))
      val (yOut, caches) = forwardModel.lModelForward(XInput = XInput, parameters = parameters)
      testMatrixValues(yOut, DenseMatrix((0.03921668, 0.70498921, 0.19734387, 0.04728177)))
      assert(caches.caches.size === 3)
    }
  }
}
