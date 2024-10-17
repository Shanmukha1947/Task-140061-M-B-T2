import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class NeuralNetworkTest {

    private MultiLayerNetwork model;

    @BeforeEach
    void setUp() {
        // Define the neural network configuration
        int inputSize = 784;     // Number of input features (MNIST: 28x28=784 pixels)
        int outputSize = 10;    // Number of output classes (MNIST: digits 0-9)
        int hiddenLayerSize = 50; // Number of neurons in the hidden layer

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(hiddenLayerSize).activation(Activation.RELU).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(hiddenLayerSize).nOut(outputSize).activation(Activation.SOFTMAX).build())
                .build();

        // Create the neural network model
        model = new MultiLayerNetwork(conf);
        model.init();
    }

    @Test
    void testModelAccuracy() throws Exception {
        // Load the MNIST test data
        DataSetIterator testDataIterator = new MnistDataSetIterator(64, false, 12345);

        // Train the model using some MNIST training data (optional, depending on your setup)
        DataSetIterator trainDataIterator = new MnistDataSetIterator(64, true, 12345);
        model.fit(trainDataIterator);  // Training the model

        // Evaluate the model on the test data
        double accuracy = model.evaluate(testDataIterator).accuracy();

        // Assert the accuracy is within an expected range
        assertEquals(0.9, accuracy, 0.05);
    }
}

