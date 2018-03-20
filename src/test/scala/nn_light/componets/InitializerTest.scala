package nn_light.componets

import breeze.linalg.{DenseMatrix, DenseVector}
import nn_light.components.{Grads, Parameters, RandomInitializer}
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class InitializerTest extends FunSuite {
  
  trait testContext {
    val initializer = RandomInitializer()
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
}
