import numpy as np
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Initialize Spark
findspark.init()

# Setup Spark configuration and context
spark_conf = pyspark.SparkConf().setAppName('Wine Quality Prediction - Training').setMaster('local[4]')

spark_context = pyspark.SparkContext(conf=spark_conf)
spark_session = SparkSession(spark_context)

# Load and preprocess the dataset
# data_frame = spark_session.read.format("csv").option("header", "true").option("sep", ";").load("s3://aws-logs-992382847407-us-east-1/EMAP/TrainingDataset.csv")
data_frame = spark_session.read.format("csv").option("header", "true").option("sep", ";").load("TrainingDataset.csv")

data_frame.printSchema()
data_frame.show(5)

# Convert data types and clean column names
for column in data_frame.columns[1:-1] + ['""""quality"""""']:
    data_frame = data_frame.withColumn(column, col(column).cast('float'))
data_frame = data_frame.withColumnRenamed('""""quality"""""', "quality")

# Extract features and labels
features = np.array(data_frame.select(data_frame.columns[1:-1]).collect())
labels = np.array(data_frame.select('quality').collect())

# Function to create labeled points
def create_labeled_points(features, labels):
    return [LabeledPoint(label, feature) for feature, label in zip(features, labels)]

# Function to convert to RDD
def convert_to_rdd(spark_context, labeled_points):
    return spark_context.parallelize(labeled_points)

labeled_points = create_labeled_points(features, labels)
labeled_points_rdd = convert_to_rdd(spark_context, labeled_points)

model_train, model_test = labeled_points_rdd.randomSplit([0.7, 0.3], seed=21)

RF = RandomForest.trainClassifier(model_train, numClasses=10, categoricalFeaturesInfo={},
                                       numTrees=21, featureSubsetStrategy="auto",
                                       impurity='gini', maxDepth=30, maxBins=32)

prediction = RF.predict(model_test.map(lambda x: x.features))
prediction_rdd = model_test.map(lambda y: y.label).zip(prediction)
prediction_df = prediction_rdd.toDF()

quality_prediction = prediction_rdd.toDF(["quality", "prediction"])
quality_prediction.show()
quality_prediction_df = quality_prediction.toPandas()

print("---------------Output-----------------")
print("Accuracy : ", accuracy_score(quality_prediction_df['quality'], quality_prediction_df['prediction']))
print("F1- score : ", f1_score(quality_prediction_df['quality'], quality_prediction_df['prediction'], average='weighted'))

test_error = prediction_rdd.filter(
    lambda y: y[0] != y[1]).count() / float(model_test.count())
print('Test Error : ' + str(test_error))
# RF.save(spark_context,"s3://aws-logs-992382847407-us-east-1/model")
RF.save(spark_context,"model")
