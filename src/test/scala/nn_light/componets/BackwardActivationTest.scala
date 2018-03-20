package nn_light.componets

import breeze.linalg.{BitVector, DenseMatrix, DenseVector}
import nn_light.components.{BackwardActivation, BackwardActivationImpl, Derivatives, LinearCache}
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class BackwardActivationTest extends FunSuite {
  
  trait SetUp {
    val backwardActivation: BackwardActivation = BackwardActivationImpl()
    
    def testMatrixValues(actual: DenseMatrix[Double], expected: DenseMatrix[Double]) = {
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
        DenseVector.fill(devs.db.length) {0.00001}
      assert(diffDb.forall(x => x))
    }
  }
}
