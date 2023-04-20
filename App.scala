package upm.bd.group15

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.types._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.mllib.evaluation.RegressionMetrics
import scala.io._
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import java.nio.file.Files
import java.nio.file.Paths

object App {

  def main(args: Array[String]) {

    println("Arrival Delay Predicting Application - Big Data")
    println("Authors: Jiayun Liu, Yuxiao Xiong and Jinglei Xu")

    //Introduce dataset file path
    var csv_file: String = ""
    while(! Files.isRegularFile(Paths.get(csv_file))){
        println("--> Introduce the absolute path of the csv file (e.g. /Users/hi/2008.csv):")
        csv_file = readLine()
        if(! Files.isRegularFile(Paths.get(csv_file))){
            println("The csv file does not exist or can not be read. Try again.")
        }
    }

    //Introduce model type
    var mlModel: Int = 0
    while(mlModel != 1 && mlModel != 2 && mlModel != 3){
      println()
      println("--> Choose a Machine Learning model for the dataset(e.g. 1): ")
      println("1 - Linear Regression")
      println("2 - Decision Trees")
      println("3 - Random Forest Trees")
      mlModel = readInt()
    }

    //Introduce if removing correlation
    var data_mod: Int = -1
    while(data_mod != 0 && data_mod != 1){
      println()
      println("--> Decide if you want to remove correlated variables or not: ")
      println("1 - Yes (Remove correlated variables)")
      println("0 - No (Do not remove correlated variables)")
      data_mod = readInt()
    }


    //Making the output less verbose
    Logger.getLogger("org").setLevel(Level.WARN)

    //App configuration and context creation
    val conf = new SparkConf().setAppName("Arrival Delay Prediction Application")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val spark = SparkSession.builder().appName("Arrival Delay Prediction Application")
        .config("spark.master", "local")
        .getOrCreate()

    //------------------------1. Loading the data---------------------------------
    // Define datatypes for the columns of dataset
    val schema = new StructType()
      .add("Year",IntegerType,true)
      .add("Month",IntegerType,true)
      .add("DayofMonth",IntegerType,true)
      .add("DayOfWeek",IntegerType,true)
      .add("DepTime",IntegerType,true)
      .add("CRSDepTime",IntegerType,true)
      .add("ArrTime",IntegerType,true)
      .add("CRSArrTime",IntegerType,true)
      .add("UniqueCarrier",StringType,true)
      .add("FlightNum",IntegerType,true)
      .add("TailNum",StringType,true)
      .add("ActualElapsedTime",DoubleType,true)
      .add("CRSElapsedTime",DoubleType,true)
      .add("AirTime",DoubleType,true)
      .add("ArrDelay",DoubleType,true)
      .add("DepDelay",DoubleType,true)
      .add("Origin",StringType,true)
      .add("Dest",StringType,true)
      .add("Distance",DoubleType,true)
      .add("TaxiIn",DoubleType,true)
      .add("TaxiOut",DoubleType,true)
      .add("Cancelled",IntegerType,true)
      .add("CancellationCode",StringType,true)
      .add("Diverted",IntegerType,true)
      .add("CarrierDelay",IntegerType,true)
      .add("WeatherDelay",IntegerType,true)
      .add("NASDelay",IntegerType,true)
      .add("SecurityDelay",IntegerType,true)
      .add("LateAircraftDelay",IntegerType,true)

    val data = spark.read.option("header", "true") // Keep the header of the dataset
        .schema(schema) // Set datatypes
        .csv(""+csv_file)
    //data.show(100)
 
    //------------------------2. Processing the data-----------------------------
    val df = data
        //Forbidden variables
        .drop("ArrTime") 
        .drop("ActualElapsedTime")
        .drop("AirTime")
        .drop("TaxiIn")
        .drop("Diverted")
        .drop("CarrierDelay")
        .drop("WeatherDelay")
        .drop("NASDelay")
        .drop("SecurityDelay")
        .drop("LateAircraftDelay") 
        //Unecessary variables
        .drop("Cancelled") 
        .drop("CancellationCode")
        .drop("UniqueCarrier")
        // remove cancelled flights
        .filter("DepDelay is not null")
        // remove NaN rows
        .na.drop()
    
    import spark.implicits._

    //Removing high-correlated variables like DepTime, CRSDepTime, CRSArrTime
    //while change DepTime to Morning, afternoon, evening, night depending on its value
    val timeTransform = udf((time: Double) => {
      time match {
        case time if (time > 500.0 && time <= 1200.0)   => "Morning"
        case time if (time > 1200.0 && time <= 1700.0) => "Afternoon"
        case time if (time > 1700.0 && time <= 2100.0) => "Evening"
        case _                                         => "Night"
      }
    })
    val df2 = if(data_mod==1){//Without correlated variables
                df.withColumn("DepTimeRange", timeTransform($"DepTime") cast "String")
                //High-correlated variables
                .drop("DepTime")
                .drop("CRSDepTime")
                .drop("CRSArrTime")
                .drop("CRSElapsedTime")
            }else{
                df
            } 

    //------------------------------3. Creating and Validating the model--------------------------  
    val categoricalVar = if(data_mod==1){//Without correlated variables
                            Array("Year","Month","DayofMonth","DayOfWeek","FlightNum",
                            "TailNum","Origin","Dest", "DepTimeRange")
                        }else{
                            Array("Year","Month","DayofMonth","DayOfWeek","FlightNum",
                            "TailNum","Origin","Dest")
                        }
    val numericVar = if(data_mod==1){//Without correlated variables
                        Array("DepDelay", "Distance", "TaxiOut")
                    }else{
                        Array("DepDelay", "Distance", "TaxiOut","DepTime","CRSDepTime",
                        "CRSArrTime","CRSElapsedTime")
                    }
    
    // Transform categorical variables
    val categoricalIndex = categoricalVar.map(i => 
        new StringIndexer().setInputCol(i).setOutputCol(i+"Idx").setHandleInvalid("skip"))
    val categoricalEncoder = categoricalVar.map(i => 
        new OneHotEncoder().setInputCol(i + "Idx").setOutputCol(i + "Enc").setDropLast(false))
    val categoricalEncoderCols : Array[String] = categoricalVar.map(i => s"${i}Enc")
    
    //Pipeline stages
    val assembler = new VectorAssembler()
        .setInputCols(categoricalEncoderCols ++ numericVar)
        .setOutputCol("features")
        .setHandleInvalid("skip")
    
    //Traning and testing ratio = 8:2
    val Array(training, test) = df2.randomSplit(Array(0.8, 0.2), seed = 68743)
    mlModel match {
        case 1 => //------------------------3.1 Linear Regression--------------------------
            val lr = new LinearRegression()
                .setFeaturesCol("features")
                .setLabelCol("ArrDelay")

            // set 5 grids
            val lrparamGrid = new ParamGridBuilder()
                .addGrid(lr.regParam, Array(0.001, 0.01, 0.1, 0.5, 1.0, 2.0))
                .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
                .addGrid(lr.maxIter, Array(1, 5, 10, 20, 50))
                .build()
    
            val lrsteps: Array[PipelineStage] = categoricalIndex ++ categoricalEncoder ++ Array(assembler, lr)
            val lrpipeline = new Pipeline().setStages(lrsteps)

            // Return the prediction of the response variable
            val lrevaluator = new RegressionEvaluator()
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction")

            //Use tran validation split to find the best of the 5 grids.
            val lrtvs = new TrainValidationSplit()
                .setEstimator(lrpipeline) // the estimator can also just be an individual model rather than a pipeline
                .setEvaluator(lrevaluator)
                .setEstimatorParamMaps(lrparamGrid)
                .setTrainRatio(0.8)
        
            val lrmodel = lrtvs.fit(training)

            val lrpredictions = lrmodel.transform(test)

            //Get the predictions and the real values
            val lrholdout = lrmodel.transform(test).select("prediction", "ArrDelay")

            
            val lrrm = new RegressionMetrics(lrholdout.rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
            
            println("RMSE: " + Math.sqrt(lrrm.meanSquaredError))
            println("R Squared: " + lrrm.r2)
            println("Explained Variance: " + lrrm.explainedVariance)
      
        case 2 => //------------------------3.2 Decision Trees--------------------------
            // Train a DecisionTree model.
            val dt = new DecisionTreeRegressor()
                .setLabelCol("ArrDelay")
                .setFeaturesCol("features")

            val dtsteps: Array[PipelineStage] = categoricalIndex ++ categoricalEncoder ++ Array(assembler, dt)
            val dtpipeline = new Pipeline().setStages(dtsteps)

            val dtmodel = dtpipeline.fit(training)

            val predictions = dtmodel.transform(test)

            val dtholdout = predictions.select("prediction", "ArrDelay")

            val dtevaluator = new RegressionEvaluator()
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction")
            
            val dtrm = new RegressionMetrics(dtholdout.rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
            
            println("RMSE: " + Math.sqrt(dtrm.meanSquaredError))
            println("R Squared: " + dtrm.r2)
            println("Explained Variance: " + dtrm.explainedVariance)
        
        case 3 => //------------------------3.3 Random Forest--------------------------
            val rf = new RandomForestRegressor()
                .setLabelCol("ArrDelay")
                .setFeaturesCol("features")

            val rfsteps: Array[PipelineStage] = categoricalIndex ++ categoricalEncoder ++ Array(assembler, rf)
            val rfpipeline = new Pipeline().setStages(rfsteps)

            // Train model.
            val rfmodel = rfpipeline.fit(training)

            // Make predictions.
            val rfpredictions = rfmodel.transform(test)
            
            val rfholdout = rfpredictions.select("prediction", "ArrDelay")

            // Select (prediction, true label) and compute test error.
            val rfevaluator = new RegressionEvaluator()
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction")

            val rfrm = new RegressionMetrics(rfholdout.rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

            println("RMSE: " + Math.sqrt(rfrm.meanSquaredError))
            println("R Squared: " + rfrm.r2)
            println("Explained Variance: " + rfrm.explainedVariance)

    }
    val model_algo = if(mlModel==1){
                        "Linear Regression"
                    }else if(mlModel==2){
                        "Decision Trees"
                    }else{
                        "Random Forest"
                    }
    val corr_vars = if(data_mod==0){
                        "without correlated variables"
                    }else{
                        "with correlated variables"
                    }
    println("-------------Above results come from running "+model_algo+" on the dataset "+corr_vars+".--------------\n")
  }
}