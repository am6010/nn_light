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
    assert(actual ===  DenseMatrix.fill(2, 3){1e8})
  }

  test("convert -Infinite to -1e4") {
    val input = DenseMatrix.fill(2, 3){Double.NegativeInfinity}
    val actual = removeInfiniteAndNans(input)
    assert(actual ===  DenseMatrix.fill(2, 3){-1e8})
  }

  test("convert NaN to zero") {
    val input = DenseMatrix.fill(2, 3){Double.NaN}
    val actual = removeInfiniteAndNans(input)
    assert(actual ===  DenseMatrix.zeros[Double](2, 3))
  }
  
  test("random permutation for 4 should return a Set from 0 ton 3") {
    val permutation = randomPermutation(4)
    assert(permutation.toSet === (0 to 3).toSet)
  }

  test("random permutation for 1000 should return a Set from 0 ton 999") {
    val permutation = randomPermutation(1000)
    assert(permutation.toSet === (0 to 999).toSet)
  }
}
