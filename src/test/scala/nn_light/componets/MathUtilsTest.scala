package nn_light.componets

import breeze.linalg.DenseMatrix
import nn_light.components.MathUtils._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class MathUtilsTest extends FunSuite {

  test("convert +Infinite to 1e4") {
    val input = DenseMatrix.fill(2, 3){Double.PositiveInfinity}
    val actual = removeInfiniteAndNans(input)
    assert(actual ===  DenseMatrix.fill(2, 3){1e4})
  }

  test("convert -Infinite to -1e4") {
    val input = DenseMatrix.fill(2, 3){Double.NegativeInfinity}
    val actual = removeInfiniteAndNans(input)
    assert(actual ===  DenseMatrix.fill(2, 3){-1e4})
  }

  test("convert NaN to zero") {
    val input = DenseMatrix.fill(2, 3){Double.NaN}
    val actual = removeInfiniteAndNans(input)
    assert(actual ===  DenseMatrix.zeros[Double](2, 3))
  }
}
