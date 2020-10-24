
import java.io.{File, PrintWriter}

import org.apache.spark.sql.{Encoder, Encoders, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics




object Tweet {
  def main(args : Array[String]) : Unit ={

    // TO LOCALLY RUN - > SPECIFY INPUT AND OUTPUT HERE

    //    val input_file = "Tweets.csv"
    //    val output_file = "output.txt"


    if (args.length != 2) {
      println("Usage: input_file output_Dir")
    }

    val input_file = args(0)
    val output_Dir =args(1)

    val writer = new PrintWriter(output_Dir+"output")

    // TO LOCALLY RUN - > UNCOMMENT THIS

    //    val spark: SparkSession = SparkSession.builder()
    //      .master("local[1]")
    //      .appName("SparkByExample")
    //      .config("spark.master", "local")
    //      .getOrCreate()

    // TO LOCALLY RUN - > COMMENT THIS

    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("SparkByExample")
      .config("spark.master", "local")
      .getOrCreate()


    val training_df = spark.read.options(Map("inferSchema"->"true","header"->"true")).csv(input_file).where("text is not null")

    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(indexer,tokenizer,remover,hashingTF,lr))

    val Array(train, test) = training_df.randomSplit(Array(0.9, 0.1))
    // val model = pipeline.fit(train)
    // val result = model.transform(test)


    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)  // Use 3+ in practice

    val cvModel = cv.fit(train)

    val results = cvModel.transform(test)
    //    writer.write(results.show().toString)

    val predictionAndLabels = results.select("prediction","label")
      .as[(Double, Double)]( Encoders.product[(Double,Double)]).rdd

    val mMetrics = new MulticlassMetrics(predictionAndLabels)
    val labels = mMetrics.labels

    writer.write("Classification Model: Logistic Regression \nConfusion matrix: \n")
    writer.write(mMetrics.confusionMatrix.toString())
    //    println(mMetrics.confusionMatrix)

    // Precision by label
    labels.foreach { l =>
      writer.write(s"\n Precision($l) = " + mMetrics.precision(l))
    }
    // Recall by label
    labels.foreach { l =>
      writer.write(s"\n Recall($l) = " + mMetrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      writer.write(s"\n FPR($l) = " + mMetrics.falsePositiveRate(l))
    }
    // F-measure by label
    labels.foreach { l =>
      writer.write(s"\n F1-Score($l) = " + mMetrics.fMeasure(l))
    }

    writer.close()


  }
}
