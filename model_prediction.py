import numpy as np
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Initialize Spark
findspark.init()

# Setup Spark configuration and context
spark_conf = pyspark.SparkConf().setAppName('Wine Quality Prediction').setMaster('local')
spark_context = pyspark.SparkContext(conf=spark_conf)
spark_session = SparkSession(spark_context)

# Load and process dataset
data_frame = spark_session.read.format("csv").option("header", "true").option("sep", ";").load("ValidationDataset.csv")
data_frame.printSchema()
data_frame.show(5)

# Convert columns to float and rename mislabeled columns
for column in data_frame.columns[1:-1] + ['""""quality"""""']:
    data_frame = data_frame.withColumn(column, col(column).cast('float'))
data_frame = data_frame.withColumnRenamed('""""quality"""""', "quality")

# Extract features and labels
feature_data = np.array(data_frame.select(data_frame.columns[1:-1]).collect())
label_data = np.array(data_frame.select('quality').collect())

# Functions to prepare the data for the model
def create_labeled_points(features, labels):
    labeled_points_list = [LabeledPoint(label, feature) for feature, label in zip(features, labels)]
    return labeled_points_list

def convert_to_rdd(spark_context, labeled_points):
    return spark_context.parallelize(labeled_points)

# Prepare data for prediction
labeled_points = create_labeled_points(feature_data, label_data)
labeled_points_rdd = convert_to_rdd(spark_context, labeled_points)

# Load trained model
random_forest_model = RandomForestModel.load(spark_context, "model")
print("Model successfully loaded.")

# Prediction
predictions = random_forest_model.predict(labeled_points_rdd.map(lambda x: x.features))
predictions_rdd = labeled_points_rdd.map(lambda y: y.label).zip(predictions)
predictions_df = predictions_rdd.toDF(["actual_quality", "predicted_quality"])
predictions_df.show(5)
predictions_pandas = predictions_df.toPandas()

# Evaluation
print("---------------Results-----------------")
accuracy = accuracy_score(predictions_pandas['actual_quality'], predictions_pandas['predicted_quality'])
f1 = f1_score(predictions_pandas['actual_quality'], predictions_pandas['predicted_quality'], average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")

# Calculate test error
test_error = predictions_rdd.filter(lambda y: y[0] != y[1]).count() / float(labeled_points_rdd.count())
print(f'Test Error: {test_error}')
