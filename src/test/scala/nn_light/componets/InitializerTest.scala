package nn_light.componets

import breeze.linalg.{DenseMatrix, DenseVector}
import nn_light.components.{Grads, HeInitializer, Parameters, RandomInitializer}
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FunSuite}
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class InitializerTest extends FunSuite {
  
  trait testContext {
    val initializer = RandomInitializer(0.01)
    def testMatrixValues(actual: DenseMatrix[Double], expected: DenseMatrix[Double]): Assertion = {
      val  dif = actual -:- expected <:< DenseMatrix.fill(actual.rows, actual.cols){0.0000001}
      assert(dif.forall(x => x))
    }
  }
  
  test("empty list as input") {
    new testContext {
      assertThrows[RuntimeException] {
        initializer.initializeParametersDeep(Seq())
      }
    }
  }

  test("only one number as input") {
    new testContext {
      assertThrows[RuntimeException] {
        initializer.initializeParametersDeep(Seq(3))
      }
    }
  }
  
  test("initilizer not return null") {
    new testContext {
      val parameters: Parameters = initializer.initializeParametersDeep(Seq(2, 3, 1))
      assert(parameters != null)
    }
  }

  test("initilizer should return 2 weights and 2 bias") {
    new testContext {
      val parameters: Parameters = initializer.initializeParametersDeep(Seq(2, 3, 1))
      assert(parameters.weights.size === 2)
      assert(parameters.bias.size === 2)
    }
  }

  test("initilizer test shapes") {
    new testContext {
      val parameters: Parameters = initializer.initializeParametersDeep(Seq(5, 4 ,3))
      val W1 = parameters.weights("W1")
      val W2 = parameters.weights("W2")
      val b1 = parameters.bias("b1")
      val b2 = parameters.bias("b2")
      
      assert(W1.rows === 4 && W1.cols === 5)
      assert(W2.rows === 3 && W2.cols === 4)
      assert(b1.length === 4)
      assert(b2.length === 3)
    }
  }
  
  test("initialise He model") {
    val parameters = HeInitializer().initializeParametersDeep(Seq(2, 4, 1))
    val W1 = parameters.weights("W1")
    val W2 = parameters.weights("W2")
    val b1 = parameters.bias("b1")
    val b2 = parameters.bias("b2")

    assert(W1.rows === 4 && W1.cols === 2)
    assert(W2.rows === 1 && W2.cols === 4)
    assert(b1.length === 4)
    assert(b2.length === 1)
  }
  
  
  test("update test") {
    new testContext {
      val parameters: Parameters = initializer.initializeParametersDeep(Seq(3 , 2))
      val dW1 = DenseMatrix((0.2, -0.1, 0.1), (-0.4, 0.8, -0.7))
      val db1 = DenseVector( 0.3, - 0.5)
      
      val grads = Grads(Map(("dW1", dW1)), Map(("db1", db1)))
      
      val newParameters: Parameters = parameters.update(grads, 0.00025)
      assert(newParameters.weights("W1") !== null)
      assert(newParameters.bias("b1") !== null)
    }
  }

  test("update test 2") {
    new testContext {
      val W1 = DenseMatrix((-0.41675785, -0.05626683, -2.1361961 ,  1.64027081),
        (-1.79343559, -0.84174737,  0.50288142, -1.24528809),
        (-1.05795222, -0.90900761,  0.55145404,  2.29220801))
      
      val W2 = DenseMatrix((-0.5961597 , -0.0191305 ,  1.17500122))
      
      val b1 = DenseVector(0.04153939, -1.11792545, 0.53905832)
      val b2 = DenseVector(-0.74787095)
      
      val parameters: Parameters = Parameters(Map(("W1", W1), ("W2",W2)), 
        Map(("b1", b1), ("b2", b2))
      
      )
      
      val dW1 = DenseMatrix((1.78862847,  0.43650985,  0.09649747, -1.8634927),
        (-0.2773882 , -0.35475898, -0.08274148, -0.62700068),
        (-0.04381817, -0.47721803, -1.31386475,  0.88462238))
      val db1 = DenseVector( 0.88131804, 1.70957306, 0.05003364)
      val dW2 = DenseMatrix((-0.40467741, -0.54535995, -1.54647732))
      val db2 = DenseVector(0.98236743)

      val grads = Grads(Map(("dW1", dW1), ("dW2", dW2)), Map(("db1", db1), ("db2", db2)))

      val newParameters: Parameters = parameters.update(grads, 0.1)
      
      testMatrixValues(newParameters.weights("W1"), DenseMatrix(
        (-0.59562069, -0.09991781, -2.14584584,  1.82662008),
        (-1.76569676, -0.80627147 , 0.51115557,-1.18258802),
        (-1.0535704,  -0.86128581 , 0.68284052,  2.20374577)))
      testMatrixValues(newParameters.weights("W2"), DenseMatrix((-0.55569196, 0.0354055, 
        1.32964895)))
      assert((newParameters.bias("b1") -:- DenseVector(-0.04659241, -1.28888275, 0.53405496))
        .forall( _ < 0.000001))
      assert((newParameters.bias("b2") -:- DenseVector(-0.84610769)).forall( _ < 0.000001))
    }
  }
}
