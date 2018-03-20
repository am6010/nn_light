package nn_light.componets

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid
import nn_light.components.{Relu, Sigmoid}
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ActivationFunctionTest extends FunSuite {

  test("Sigmoid function activate single on empty input") {
    val activations = Sigmoid().activate(DenseMatrix.zeros(0, 0))
    assert(activations.rows === 0)
  }
  
  test("Sigmoid function activate single") {
    val activations = Sigmoid().activate(DenseMatrix((0.8, 0.7, 1.3, -2.0)))
    assert(activations.cols === 4)
    val delta = 0.00000001
    assert(math.abs(activations(0, 0) - (1 / (1 + math.exp(-0.8)))) < delta)
    assert(math.abs(activations(0, 1) - (1 / (1 + math.exp(-0.7)))) < delta)
    assert(math.abs(activations(0, 2) - (1 / (1 + math.exp(-1.3)))) < delta)
    assert(math.abs(activations(0, 3) - (1 / (1 + math.exp(2)))) < delta)
  }
  
  test("Sigmoid derivative") {
    val input = DenseMatrix((0.8, 0.7, 1.3, -2.0))
    val derivatives = Sigmoid().firstDerivative(input)
    val delta = 0.00000001
    val expected = input.map {e => 
      val sig = sigmoid(e)
      sig * (1 - sig)
    }
    assert(derivatives === expected)
  }
  
  test("Relu activation function on empty vector") {
    val activations = Relu().activate(DenseMatrix.zeros(0, 0))
    assert(activations.rows === 0)
  }

  test("Relu function activate single ") {
    val activations = Relu().activate(DenseMatrix((0.8, 0.7, 1.3, -2.0)))
    assert(activations.cols === 4)
    val delta = 0.00000001
    assert(math.abs(activations(0, 0) - 0.8) < delta)
    assert(math.abs(activations(0, 1) - 0.7) < delta)
    assert(math.abs(activations(0, 2) - 1.3) < delta)
    assert(math.abs(activations(0, 3) - 0.0) < delta)
  }
  
  test("relu derivative test") {
    val derivatives = Relu().firstDerivative(DenseMatrix((0.8, 0.7, 1.3, -2.0)))
    assert(derivatives === DenseMatrix((1.0, 1.0, 1.0, 0.0)))
  }
}
