package nn_light.componets

import breeze.linalg.{DenseMatrix, DenseVector}
import nn_light.components._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class GradientCheckerTest extends FunSuite {

  
  test("simple grad check") {
    val layers = Seq(4, 5, 3, 1)

    val W1 = ("W1", DenseMatrix((-0.3224172 , -0.38405435,  1.13376944, -1.09989127),
      (-0.17242821, -0.87785842,  0.04221375,  0.58281521),
      (-1.10061918,  1.14472371,  0.90159072,  0.50249434),
      ( 0.90085595, -0.68372786, -0.12289023, -0.93576943),
      (-0.26788808,  0.53035547, -0.69166075, -0.39675353)))
    val b1 = ("b1", DenseVector(-0.6871727 , -0.84520564, -0.67124613, -0.0126646, -1.11731035))
    val W2 = ("W2", DenseMatrix(( 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896),
      (-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548),
      ( 2.10025514,  0.12015895,  0.61720311,  0.30017032, -0.35224985)))
    val b2 = ("b2", DenseVector(-1.1425182, -0.34934272, -0.20889423))
    val W3 = ("W3", DenseMatrix((0.58662319,  0.83898341,  0.93110208)))
    val b3 = ("b3", DenseVector(0.28558733))
    val parameters = Parameters(Map(W1, W2, W3), Map(b1, b2, b3))

    val forward = new LForwardModel()
    val entropy = EntropyCostFunction()
    val backward = new BackwardActivationImpl()

    val X = DenseMatrix((1.62434536, -0.61175641, -0.52817175),
      (-1.07296862,  0.86540763, -2.3015387 ),
      (1.74481176, -0.7612069, 0.3190391),
      (-0.24937038,  1.46210794, -2.06014071))

    val Y = DenseMatrix((1.0, 1.0, 0.0))

    val difference = GradientChecker.gradientChecking(X, Y, layers, parameters, forward, entropy,
      backward)
    
    assert(difference < 1e-7)
  }

  test("simple grad check L2") {
    val layers = Seq(4, 5, 3, 1)

    val W1 = ("W1", DenseMatrix((-0.3224172 , -0.38405435,  1.13376944, -1.09989127),
      (-0.17242821, -0.87785842,  0.04221375,  0.58281521),
      (-1.10061918,  1.14472371,  0.90159072,  0.50249434),
      ( 0.90085595, -0.68372786, -0.12289023, -0.93576943),
      (-0.26788808,  0.53035547, -0.69166075, -0.39675353)))
    val b1 = ("b1", DenseVector(-0.6871727 , -0.84520564, -0.67124613, -0.0126646, -1.11731035))
    val W2 = ("W2", DenseMatrix(( 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896),
      (-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548),
      ( 2.10025514,  0.12015895,  0.61720311,  0.30017032, -0.35224985)))
    val b2 = ("b2", DenseVector(-1.1425182, -0.34934272, -0.20889423))
    val W3 = ("W3", DenseMatrix((0.58662319,  0.83898341,  0.93110208)))
    val b3 = ("b3", DenseVector(0.28558733))
    val parameters = Parameters(Map(W1, W2, W3), Map(b1, b2, b3))

    val lambda = 0.9
    val forward = new LForwardModel()
    val entropy = EntropyCostFunctionL2(lambda)
    val backward = new BackwardActivationL2Impl(lambda)

    val X = DenseMatrix((1.62434536, -0.61175641, -0.52817175),
      (-1.07296862,  0.86540763, -2.3015387 ),
      (1.74481176, -0.7612069, 0.3190391),
      (-0.24937038,  1.46210794, -2.06014071))

    val Y = DenseMatrix((1.0, 1.0, 0.0))

    val difference = GradientChecker.gradientChecking(X, Y, layers, parameters, forward, entropy,
      backward)

    assert(difference < 1e-7)
  }
  
}
