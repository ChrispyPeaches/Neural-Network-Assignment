import java.io.IOException;

/**
 * Name: Chris Perry
 * Student #: 10327025
 * Date Turned In: 10-25-23
 * Assignment #: 2 Pt 2
 * Description:
 * The program allows the user to train & test a 784 by 15 by 10 neural network.
 * - Training uses stochastic gradient descent
 */
public class Main {
    /**
     * Program entry point
     * @param args The arguments given at program start
     * @throws IOException Thrown if there's an issue creating the output file
     */
    public static void main(String[] args) throws IOException {
        // Initialize engine
        NeuralEngine engine = new NeuralEngine(new int[]{784, 15, 10}, 6000, 10);

        // Display initial message

        // Handle input
        while (true) {
            switch (IOHelper.GetEngineModeFromInput(engine.CurrentBiasVectors.size() == (engine.LayerSizes.length - 1))) {
                case TrainWithRandomWeights -> engine.TrainEngine(3, 1, IOHelper.DataSetType.Training);
                case LoadPreTrainedNetwork -> IOHelper.LoadWeightsFromFile(engine);
                case TrainingDataAccuracyDisplay -> engine.DemoEngine(IOHelper.DataSetType.Training, IOHelper.OutputType.Accuracy);
                case TestingDataAccuracyDisplay -> engine.DemoEngine(IOHelper.DataSetType.Testing, IOHelper.OutputType.Accuracy);
                case RunTestingDataAndShowImages -> engine.DemoEngine(IOHelper.DataSetType.Testing, IOHelper.OutputType.AllImages);
                case DisplayMisclassifiedTestingImages -> engine.DemoEngine(IOHelper.DataSetType.Testing, IOHelper.OutputType.MisclassifiedImages);
                case SaveNetworkState -> IOHelper.SaveWeightsToFile(engine);
                case ExitProgram -> System.exit(0);
            }
        }
    }
}
