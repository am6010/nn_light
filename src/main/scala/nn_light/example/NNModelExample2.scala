package nn_light.example

import breeze.linalg.{*, DenseMatrix, sum}
import breeze.numerics.{abs, sin}
import breeze.stats.distributions.Rand
import nn_light.components._

object NNModelExample2 extends App {
  val numOfFeatures = 2
  val trainX = DenseMatrix.rand(numOfFeatures, 1000, Rand.gaussian)
  val testX = DenseMatrix.rand(numOfFeatures, 64, Rand.gaussian)
  val sinTrainX = sin(trainX * (2 * math.Pi))
  val sinTestX = sin(testX * (2 * math.Pi))
  
  val trainY = sum(sinTrainX(::, *)).inner
    .mapValues(x => if (x > 0.0) 1.0 else 0.0).toDenseMatrix

  val testY = sum(sinTestX(::, *)).inner
    .mapValues(x => if (x > 0.0) 1.0 else 0.0).toDenseMatrix
  
  val context = SimpleNNContext(Seq(trainX.rows, 30, 20, 7, 5, 1), 1000,
    HeInitializer(),
    new LForwardModel(),
    new EntropyCostFunction(),
    new BackwardActivationImpl(),
    new RandomMiniBatchWithADAMOptimizer(64, 10000, 0.012, 0.9, 0.999, 1e-8, 1e-5))
    // new MomentumRandomMiniBatchGradientDescentOptimizer(64, 10000, 0.012, 0.9, 1e-8))
    // new RandomMiniBatchGradientDescentOptimizer(64, 5000, 0.012, 0.00001))
    //new GradientDescentOptimizer(40000, 0.08))

  val nn = DeepNN(context)

  val costs = nn.train(trainX, trainY)
  
  println(costs.min)

  val trainPred = nn.predict(trainX)
  val trainAcc = 1.0 - (sum(abs(nn.predict(trainX) - trainY)) / trainY.cols)
  println(s"Train accuracy -> $trainAcc")

  val pred = nn.predict(testX)
  val testAcc = 1 - (sum(abs(pred - testY)) / testY.cols)
  println(s"Test accuracy -> $testAcc")
}
