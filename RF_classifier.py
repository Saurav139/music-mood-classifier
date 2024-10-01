
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import when
# 1. Initialize Spark Session
spark = SparkSession.builder.appName("RandomForestClassificationExample").getOrCreate()

# 2. Read the Data
df = spark.read.csv("tracks_features_50000.csv", header=True, inferSchema=True)

# 3. Preprocess the Data
# Specify the columns for features and the target variable
feature_columns = [col for col in df.columns if col != 'label']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(df)

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 4. Build and Train the Model
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label",numTrees=60,maxDepth=30)
pipeline = Pipeline(stages=[rf_classifier])
model = pipeline.fit(train_data)

# 5. Evaluate the Model
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy on test data = {accuracy}")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="Weightedprecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="Weightedrecall")
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
print(f"Precision on test data = {precision}")
print(f"Recall on test data = {recall}")
# Stop the Spark session
spark.stop()
