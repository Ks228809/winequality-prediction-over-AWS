package wineClassification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class LogisticRegression {

	public static void main(String[] args) {

		SparkConf sparkConf = new SparkConf().setAppName("WineClassification");
		JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

		String trainingDataPath = "s3n://wine-classification/TrainingDataset.csv";
		JavaRDD<String> trainingData = sparkContext.textFile(trainingDataPath);

		String headerTrain = trainingData.first();
		JavaRDD<String> filteredTrainingData = trainingData.filter(s -> !s.contains(headerTrain));

		JavaRDD<LabeledPoint> parsedTrainingData = filteredTrainingData.map(line -> {
			String[] tokens = line.split(";");
			double[] features = new double[tokens.length - 1];
			for (int i = 0; i < features.length; i++) {
				features[i] = Double.parseDouble(tokens[i]);
			}
			return new LabeledPoint(Double.parseDouble(tokens[tokens.length - 1]), Vectors.dense(features));
		});

		parsedTrainingData.cache();

		String validationDataPath = "s3n://wine-classification/ValidationDataset.csv";
		JavaRDD<String> validationData = sparkContext.textFile(validationDataPath);

		String headerValidation = validationData.first();
		JavaRDD<String> filteredValidationData = validationData.filter(s -> !s.contains(headerValidation));

		JavaRDD<LabeledPoint> parsedValidationData = filteredValidationData.map(line -> {
			String[] tokens = line.split(";");
			double[] features = new double[tokens.length - 1];
			for (int i = 0; i < features.length; i++) {
				features[i] = Double.parseDouble(tokens[i]);
			}
			return new LabeledPoint(Double.parseDouble(tokens[tokens.length - 1]), Vectors.dense(features));
		});

		parsedValidationData.cache();

		LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
				.setNumClasses(10)
				.run(parsedTrainingData.rdd());

		JavaPairRDD<Object, Object> predictionAndLabels = parsedValidationData.mapToPair(
				point -> new Tuple2<>(model.predict(point.features()), point.label())
		);

		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

		double accuracy = metrics.accuracy();
		System.out.printf("\n------------------------\nValidation Accuracy: %.2f%%\n------------------------\n", accuracy * 100);

		double fMeasure = metrics.weightedFMeasure();
		System.out.printf("\n------------------------\nValidation F Measure: %.2f\n------------------------\n", fMeasure);

		model.save(sparkContext.sc(), "s3n://wine-classification/LogisticRegressionModel");
		sparkContext.stop();
	}
}
