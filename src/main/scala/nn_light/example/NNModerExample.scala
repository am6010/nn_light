package nn_light.example

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.abs
import nn_light.components._

object NNModerExample extends App {
  
  
  def loadFile(filename: String): DenseMatrix[Double] = {
    val trainXArr = scala.io.Source
      .fromFile(s"dataset/$filename")
      .getLines()
      .map(_.split(",").map(_.toDouble)).toArray

    val m = trainXArr(0).length
    val nX = trainXArr.length
    new DenseMatrix(nX, m, trainXArr.flatten)
  }
  // load train data
  
  val trainX = loadFile("train_x.csv")
  val trainY = loadFile("train_y.csv")
  val testX = loadFile("test_x.csv")
  val testY = loadFile("test_y.csv")
  
  val newTrainX = trainX.copy
  newTrainX(::, 0 until  testX.cols) := testX
  val newTrainY = trainY.copy
  newTrainY(::, 0 until  testY.cols) := testY
  
  val newTestX = trainX(::, 0 until testX.cols)
  val newTestY = trainY(::, 0 until testY.cols)

  val lambda = 2.7
  val context = SimpleNNContext(Seq(12288, 20, 7, 5, 1), 0.0075, 1000, 
    RandomInitializer(0.01),
    new LForwardModel(),
    new EntropyCostFunctionL2(lambda),
    new BackwardActivationL2Impl(lambda),
    new GradientDescentOptimizer(80500))
  
  val nn = DeepNN(context)
  
  // nn.train(DenseMatrix.horzcat(trainX, testX), DenseMatrix.horzcat(trainY, testY))
  nn.train(newTrainX, newTrainY)
  
  val trainPred = nn.predict(trainX)
  //println(trainY.toArray.foldLeft("")((s,x) => s + "%1.3f " format x))
  //println(trainPred.toArray.foldLeft("")((s,x) => s + "%1.3f " format x))

  val trainAcc = 1.0 - (sum(abs(nn.predict(newTrainX) -:- newTrainY)) / newTrainY.cols)
  
  println(s"Train accuracy -> $trainAcc")
  
  val pred = nn.predict(newTestX)
  
  // println(testY.toArray.foldLeft("")((s,x) => s + s"$x "))
  // println(pred.toArray.foldLeft("")((s,x) => s + "%1.3f " format x))
  val testAcc = 1 - (sum(abs(pred -:- newTestY)) / newTestY.cols)
  println(s"Test accuracy -> $testAcc")
}
