import argparse
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, mean, stddev
from pyspark.sql.types import DoubleType, IntegerType, ShortType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from math import sqrt
import time
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.feature import HashingTF
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def preprocessData(df):
    #ensure valence is a DoubleType and handle non-numeric values
    df = df.withColumn("valence", df["valence"].cast(DoubleType()))

    # Drop rows where valence is null
    df = df.filter(df["valence"].isNotNull())

    # Convert valence to 0 or +1
    valence_to_label = udf(lambda x: 1 if x > 0.5 else 0, ShortType())

    df = df.withColumn("label", valence_to_label(df["valence"]))

    numeric_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'duration_ms']
    for feature in numeric_features:
        df = df.withColumn(feature, col(feature).cast(DoubleType()))
        feature_mean = df.select(mean(col(feature))).collect()[0][0]
        feature_stddev = df.select(stddev(col(feature))).collect()[0][0]
        df = df.withColumn(feature, (col(feature) - feature_mean) / feature_stddev)

    categorical_features = ['explicit', 'key', 'mode', 'time_signature', 'year']
    for feature in categorical_features:
        df = StringIndexer(inputCol=feature, outputCol=feature + "_index").fit(df).transform(df)

    assembler = VectorAssembler(inputCols=numeric_features + [f + "_index" for f in categorical_features], outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    def to_sparse_vector(row):
        return SparseVector(len(row.features), [(i, v) for i, v in enumerate(row.features) if v != 0])

    df = df.rdd.map(lambda row: (row.label, to_sparse_vector(row)))
    df = spark.createDataFrame(df, ["label", "features"])


    return df


def readDataRDD(input_file, spark_session):
    df = spark_session.read.csv(input_file, header=True, inferSchema=True).limit(50000)
    start_pre = time.time()
    df = preprocessData(df)
    end_pre = time.time()

    print(f"preprocessing time: {end_pre - start_pre} seconds")

    # Convert DataFrame to LabeledPoint RDD using SparseVectors
    def mapRowToLabeledPoint(row):
        return LabeledPoint(row.label, SparseVector(len(row.features), [(i, v) for i, v in enumerate(row.features) if v != 0]))

    return df.rdd.map(mapRowToLabeledPoint)


def trainAndEvaluate(trainRDD, testRDD, maxDepth, maxBins):
    start_train = time.time()
    model = DecisionTree.trainClassifier(trainRDD, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=maxDepth, maxBins=maxBins)#    predictions = model.predict(testRDD.map(lambda x: x.features))
    end_train = time.time()

    print(f"Training time: {end_train - start_train} seconds")
    start_predict = time.time()
    predictions = model.predict(testRDD.map(lambda x: x.features))
    labelsAndPredictions = testRDD.map(lambda lp: lp.label).zip(predictions)
    end_predict = time.time()
    print(f"Prediction time: {end_predict - start_predict} seconds")

    accuracy = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(testRDD.count())
    print(f"Model accuracy: {accuracy}")
    start_auc = time.time()
    metrics = BinaryClassificationMetrics(labelsAndPredictions)
    auc_score = metrics.areaUnderROC
    end_auc = time.time()
    print(f"AUC Score calculation time: {end_auc - start_auc} seconds")
    print(f"AUC Score: {auc_score}")

    return model


def generateKFolds(dataRDD, k):
    folds = dataRDD.randomSplit([1.0 / k] * k, seed=42)
    return [(dataRDD.context.union([fold for j, fold in enumerate(folds) if j != i]), folds[i]) for i in range(k)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Decision Tree Classifier with Preprocessing.')
    parser.add_argument('--input', default='tracks_features.csv', help='Input CSV file containing the dataset')
    parser.add_argument('--maxDepth', type=int, default=7, help='Maximum depth of the tree')
    parser.add_argument('--maxBins', type=int, default=32, help='Maximum number of bins for the decision tree')
    parser.add_argument('--N', type=int, default=5, help='Number of partitions for data parallelism')
    parser.add_argument('--k', type=int, default=4, help='Number of folds for cross-validation')
    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Decision Tree Classifier')
    spark = SparkSession.builder.getOrCreate()
    start_time = time.time()

    dataRDD = readDataRDD(args.input, spark).repartition(args.N).cache()
    num_positive_labels = dataRDD.filter(lambda lp: lp.label == 1).count()
    num_negative_labels = dataRDD.filter(lambda lp: lp.label == 0).count()

    print(f"Number of positive labels: {num_positive_labels}")
    print(f"Number of negative labels: {num_negative_labels}")


    #initialize lists to store evaluation metrics
    cross_val_scores = []
    mse = []
    mae = []
    rmse = []
    accuracy = []
    precision = []
    recall = []
    explained_variance = []

    folds = generateKFolds(dataRDD, args.k)
    for i, (trainRDD, testRDD) in enumerate(folds):
        print(f"Training on fold {i+1}")
        model = trainAndEvaluate(trainRDD, testRDD, args.maxDepth, args.maxBins)

        #evaluation on the test set
        predictions = model.predict(testRDD.map(lambda x: x.features))
        labelsAndPredictions = testRDD.map(lambda lp: lp.label).zip(predictions)

        #calculate evaluation metrics
        num_samples = testRDD.count()
        correct_predictions = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count()
        accuracy_fold = correct_predictions / float(num_samples)
        precision_fold = labelsAndPredictions.filter(lambda lp: lp[0] == 1 and lp[1] == 1).count() / labelsAndPredictions.filter(lambda lp: lp[1] == 1).count()
        recall_fold = labelsAndPredictions.filter(lambda lp: lp[0] == 1 and lp[1] == 1).count() / labelsAndPredictions.filter(lambda lp: lp[0] == 1).count()

        #append fold results to lists
        cross_val_scores.append(accuracy_fold)
        accuracy.append(accuracy_fold)
        precision.append(precision_fold)
        recall.append(recall_fold)

    mean_cross_val_scores = sum(cross_val_scores) / args.k
    mean_accuracy = sum(accuracy) / args.k
    mean_precision = sum(precision) / args.k
    mean_recall = sum(recall) / args.k

    print("Cross-Validation Scores: ", cross_val_scores)
    print("Accuracy: ", mean_accuracy)
    print("Precision: ", mean_precision)
    print("Recall: ", mean_recall)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time} seconds")
    
    spark.stop()

