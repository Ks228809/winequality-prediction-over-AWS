package wineClassification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class LogisticRegressionPrediction {

	public static void main(String[] args) {

		// Configure Spark
		SparkConf sparkConfig = new SparkConf()
				.setAppName("WineClassification")
				.setMaster("local")
				.set("spark.testing.memory", "2147480000");

		JavaSparkContext sparkContext = new JavaSparkContext(sparkConfig);

		// Load test data
		String testDataPath = args[0];
		JavaRDD<String> testData = sparkContext.textFile(testDataPath);

		// Filter out the header
		String header = testData.first();
		JavaRDD<String> filteredTestData = testData.filter(line -> !line.equals(header));

		// Parse data into LabeledPoint
		JavaRDD<LabeledPoint> parsedTestData = filteredTestData.map(line -> {
			String[] values = line.split(";");
			double[] features = new double[values.length - 1];
			for (int i = 0; i < features.length; i++) {
				features[i] = Double.parseDouble(values[i]);
			}
			return new LabeledPoint(Double.parseDouble(values[values.length - 1]), Vectors.dense(features));
		});

		parsedTestData.cache();

		// Load model
		LogisticRegressionModel model = LogisticRegressionModel.load(sparkContext.sc(), args[1]);

		// Predict and label
		JavaPairRDD<Object, Object> predictionAndLabels = parsedTestData.mapToPair(
				point -> new Tuple2<>(model.predict(point.features()), point.label())
		);

		// Evaluate model
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		double fMeasure = metrics.weightedFMeasure();

		// Output results
		System.out.printf("\n----------------------------------------\n");
		System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
		System.out.printf("----------------------------------------\n");

		System.out.printf("\n----------------------------------------\n");
		System.out.printf("F Measure: %.2f\n", fMeasure);
		System.out.printf("----------------------------------------\n");

		sparkContext.stop();
	}
}
