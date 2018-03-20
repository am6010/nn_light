package nn_light.componets

import breeze.linalg.{BitVector, DenseMatrix, DenseVector, sum}
import nn_light.components._
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FunSuite}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class BackwardActivationTest extends FunSuite {
  
  trait SetUp {
    val backwardActivation: BackwardActivation = BackwardActivationImpl()
    
    def testMatrixValues(actual: DenseMatrix[Double], expected: DenseMatrix[Double]): Assertion = {
      val  dif = actual -:- expected <:< DenseMatrix.fill(actual.rows, actual.cols){0.0000001}
      assert(dif.forall(x => x))
    }
  }
  
  test("linearBackward test") {
    new SetUp {
      val dZ = DenseMatrix((1.62434536, -0.61175641))
      val aPrev =DenseMatrix(
        (-0.52817175, -1.07296862), 
        (0.86540763, -2.3015387), 
        (1.74481176, -0.7612069))
      
      val w = DenseMatrix((0.3190391 , -0.24937038,  1.46210794))
      val b = DenseVector(-2.06014071)
      val cache = LinearCache(aPrev, w, b)
      val devs: Derivatives = backwardActivation.linearBackward(dZ, cache)
      
      testMatrixValues(devs.dAPrev, DenseMatrix((0.51822968, -0.19517421),
        (-0.40506361, 0.15255393),
        (2.37496825, -0.89445391)))
      
      testMatrixValues(devs.dW, DenseMatrix((-0.10076895, 1.40685096, 1.64992505)))
      
      val diffDb: BitVector = devs.db -:- DenseVector(0.50629448) <:<
        DenseVector.fill(devs.db.length) {0.000001}
      assert(diffDb.forall(x => x))
    }
  }
  
  test("linearActivationBackward test sigmoid") {
    new SetUp {
      val dAL = DenseMatrix((-0.41675785, -0.05626683))
      val aPrev =DenseMatrix(
        (-2.1361961 ,  1.64027081),
        (-1.79343559, -0.84174737),
        ( 0.50288142, -1.24528809))

      val w = DenseMatrix((-1.05795222, -0.90900761,  0.55145404))
      val b = DenseVector(2.29220801)
      val cache = LinearCache(aPrev, w, b)

      val activations = DenseMatrix((0.04153939, -1.11792545))
      val activationCache = ActivationCache(activations)
      val derivatives: Derivatives = backwardActivation.linearActivationBackward(dAL, 
        (cache, activationCache), Sigmoid())
      
      testMatrixValues(derivatives.dAPrev, DenseMatrix(( 0.11017994,  0.01105339),
        (0.09466817,  0.00949723),
        (-0.05743092, -0.00576154)))

      testMatrixValues(derivatives.dW, DenseMatrix(( 0.10266786,  0.09778551, -0.01968084)))
      assert(math.abs(derivatives.db(0) - (-0.05729622)) < 0.0000001)
    }
  }

  test("linearActivationBackward test relu") {
    new SetUp {
      val dAL = DenseMatrix((-0.41675785, -0.05626683))
      val aPrev =DenseMatrix(
        (-2.1361961 ,  1.64027081),
        (-1.79343559, -0.84174737),
        ( 0.50288142, -1.24528809))

      val w = DenseMatrix((-1.05795222, -0.90900761,  0.55145404))
      val b = DenseVector(2.29220801)
      val cache = LinearCache(aPrev, w, b)

      val activations = DenseMatrix((0.04153939, -1.11792545))
      val activationCache = ActivationCache(activations)
      val derivatives: Derivatives = backwardActivation.linearActivationBackward(dAL,
        (cache, activationCache), Relu())

      testMatrixValues(derivatives.dAPrev, DenseMatrix(( 0.44090989, 0.0), 
        (0.37883606, 0.0 ),
        (-0.2298228, 0.0)))

      testMatrixValues(derivatives.dW, DenseMatrix(( 0.44513824,  0.37371418, -0.10478989)))
      assert(math.abs(derivatives.db(0) - (-0.20837892)) < 0.0000001)
    }
  }
  
  test("lModelBackward test") {
    new SetUp {
      val AL = DenseMatrix((1.78862847, 0.43650985))
      val y =DenseMatrix((1.0, 0.0))
      val cache = Cache(Seq(
        (LinearCache(DenseMatrix(
          (0.09649747, -1.8634927 ), 
          (-0.2773882 , -0.35475898),
          (-0.08274148, -0.62700068),
          (-0.04381817, -0.47721803)), 
          DenseMatrix(
            (-1.31386475,  0.88462238,  0.88131804,  1.70957306), 
            (0.05003364, -0.40467741, -0.54535995, -1.54647732), 
            (0.98236743, -1.10106763, -1.18504653, -0.2056499 )),
          DenseVector(1.48614836, 0.23671627, -1.02378514)), ActivationCache(DenseMatrix(
          (-0.7129932 ,  0.62524497),
          (-0.16051336, -0.76883635),
          (-0.23003072,  0.74505627)))),
          (LinearCache(DenseMatrix((1.97611078, -1.24412333),
            (-0.62641691, -0.80376609),
            (-2.41908317, -0.92379202)), 
            DenseMatrix((-1.02387576,  1.12397796, -0.13191423)), 
            DenseVector(-1.62328545)), 
            ActivationCache(DenseMatrix((0.64667545, -0.35627076))))
      ))
      
      val grads: Grads = backwardActivation.lModelBackward(AL, y, cache)
      
      testMatrixValues(grads.matrices("dA1"), DenseMatrix(
        ( 0.12913162, -0.44014127), 
        (-0.14175655, 0.48317296), 
        (0.01663708, -0.05670698)))
      
      testMatrixValues(grads.matrices("dW1"), DenseMatrix(
        (0.41010002, 0.07807203, 0.13798444, 0.10502167), 
        (0.0, 0.0, 0.0, 0.0), 
        (0.05283652, 0.01005865, 0.01777766, 0.0135308)
      ))
      
      val sumDif = sum(grads.vectors("db1") -:- DenseVector(-0.22007063, 0.0, -0.02835349))
      assert(sumDif < 0.000001) 
    }
  }
}
