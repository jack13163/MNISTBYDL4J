package cn.rocket;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainCNN {
	private static Logger log = LoggerFactory.getLogger(TrainCNN.class);

	public static void main(String[] args) throws IOException {
		// 图片的长宽和通道数
		final int numRows = 28;
		final int numColumns = 28;
		final int channels = 1;

		int outputNum = 10; // 输出类别个数
		int batchSize = 128; // 每一批的大小
		int rngSeed = 123; // 随机数种子
		int numEpochs = 30; // 训练次数

		// 数据预处理
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

		log.info("配置模型....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed).l2(0.0005)
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Nesterovs(0.006, 0.9)).list()
				.layer(0,
						new ConvolutionLayer.Builder(5, 5).nIn(channels).stride(1, 1).nOut(20)
								.activation(Activation.IDENTITY).build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(2,
						new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(50).activation(Activation.IDENTITY)
								.build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
								.activation(Activation.SOFTMAX).build())
				.setInputType(InputType.convolutionalFlat(numRows, numColumns, channels))
				.backpropType(BackpropType.Standard).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		model.init();

		// 设置输出信息的频率
		model.setListeners(new ScoreIterationListener(1));

		log.info("训练模型....");
		model.fit(mnistTrain, numEpochs);

		log.info("评估模型....");
		Evaluation eval = model.evaluate(mnistTest);
		log.info(eval.stats());

		// 保存训练好的模型
		File locationToSave = new File("模型.zip");
		ModelSerializer.writeModel(model, locationToSave, false);

	}
}
