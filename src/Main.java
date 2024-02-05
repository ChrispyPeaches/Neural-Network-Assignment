import java.io.IOException;

/**
 * Name: Chris Perry
 * Description:
 * The program allows the user to train & test a 784 by 15 by 10 neural network using datasets from MNIST.
 * - The user has the option to:
 *      - Train a network using stochastic gradient descent
 *      - Test with ASCII output
 *          - For all test cases or
 *          - Only for misclassified test cases
 *      - Test with Accuracy results output for both the training and testing datasets
 *      - Save and load a network's state
 *      - Exit the program
 * - Note:
 *     - mnist_train.csv and mnist_test.csv files are expected to be in the directory of the program being executed
 *     - weights files are stored and read from the directory of the program being executed
 */

public class Main {
    /**
     * Program entry point
     * @param args The arguments given at program start
     * @throws IOException Thrown if there's an issue creating the output file
     */
    public static void main(String[] args) throws IOException {
        // Initialize engine
        NeuralEngine engine = new NeuralEngine(new int[]{784, 100, 10}, 10);

        // Handle input
        while (true) {
            switch (IOHelper.GetEngineModeFromInput(engine.CurrentBiasVectors.size() == (engine.LayerSizes.length - 1))) {
                case TrainWithRandomWeights -> engine.TrainEngine(3, 5, IOHelper.DataSetType.Training);
                case LoadPreTrainedNetwork -> IOHelper.LoadWeightsAndBiasesFromFile(engine);
                case TrainingDataAccuracyDisplay -> engine.DemoEngine(IOHelper.DataSetType.Training, IOHelper.OutputType.Accuracy);
                case TestingDataAccuracyDisplay -> engine.DemoEngine(IOHelper.DataSetType.Testing, IOHelper.OutputType.Accuracy);
                case RunTestingDataAndShowImages -> engine.DemoEngine(IOHelper.DataSetType.Testing, IOHelper.OutputType.AllImages);
                case DisplayMisclassifiedTestingImages -> engine.DemoEngine(IOHelper.DataSetType.Testing, IOHelper.OutputType.MisclassifiedImages);
                case SaveNetworkState -> IOHelper.SaveWeightsAndBiasesToFile(engine);
                case ExitProgram -> System.exit(0);
            }
        }
    }
}
