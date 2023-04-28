# Big Data Subject of Msc. in Data Science UPM

Practical work

How to run the app

The application will run locally, so to run the application, the local machine must have JDK8+
and SBT installed.

1. Create an empty scala app called 'App' and change the App.scala and build.sbt files

2. Open the terminal and change current directory to the folder app

3. (Optional) If the local machine doesn’t have the 3.3.0 version of spark-sql, spark-mllib or sparkcore
libraries, introduce following command in terminal:
```
sbt package>
```

4. Run the application by introducing:
```
spark-submit --master local --class upm.bd.group15.App target/scala-2.12/arrivaldelay_2.12-1.0.0.jar
```
in the terminal.

5. When the running begins, introduce the absolute path of the dataset to proceed, the
machine learning regression model and whether using high correlated variables or not
 (Linear Regression model takes around 1 hour to run for about 2.4 million
rows of data, Decision Tree and Random Forest need less time.)

6. The results will show the three metrics used to evaluate the models(R2, RMSE and explained
variance).
