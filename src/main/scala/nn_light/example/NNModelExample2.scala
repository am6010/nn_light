package nn_light.example

import breeze.linalg.{*, DenseMatrix, sum}
import breeze.numerics.{abs, log, sin}
import breeze.stats.distributions.Rand
import nn_light.components._

object NNModelExample2 extends App {
  val numOfFeatures = 64
  val trainX = DenseMatrix.rand(numOfFeatures, 20024, Rand.gaussian)
  val testX = DenseMatrix.rand(numOfFeatures, 64, Rand.gaussian)
  val sinTrainX = sin(trainX * (2 * math.Pi))
  val sinTestX = sin(testX * (2 * math.Pi))
  
  val trainY = sum(sinTrainX(::, *)).inner
    .mapValues(x => if (x > 0.0) 1.0 else 0.0).toDenseMatrix

  val testY = sum(sinTestX(::, *)).inner
    .mapValues(x => if (x > 0.0) 1.0 else 0.0).toDenseMatrix
  
  val context = SimpleNNContext(Seq(trainX.rows, 20, 7, 5, 1),  0.3 , 1000,
    HeInitializer(),
    new LForwardModel(),
    new EntropyCostFunction(),
    new BackwardActivationImpl(),
    new GradientDescentOptimizer(20000, 0.01))

  val nn = DeepNN(context)

  nn.train(trainX, trainY)

  val trainPred = nn.predict(trainX)
  val trainAcc = 1.0 - (sum(abs(nn.predict(trainX) - trainY)) / trainY.cols)
  println(s"Train accuracy -> $trainAcc")

  val pred = nn.predict(testX)
  val testAcc = 1 - (sum(abs(pred - testY)) / testY.cols)
  println(s"Test accuracy -> $testAcc")
}
