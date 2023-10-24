import java.io.IOException;

public class Main {
    /**
     * Program entry point
     * @param args The arguments given at program start
     * @throws IOException Thrown if there's an issue creating the output file
     */
    public static void main(String[] args) throws IOException {
        // Initialize engine
        NeuralEngine engine = new NeuralEngine(new int[]{784, 15, 10}, 60000, 10);

        // Handle input
        while (true) {
            switch (IOHelper.GetEngineModeFromInput(engine.Seed != null)) {
                case TrainWithRandomWeights -> engine.TrainNetwork(3, 30);
                case LoadPreTrainedNetwork -> IOHelper.LoadWeightsFromFile(engine);
                case SaveNetworkState -> IOHelper.SaveWeightsToFile(engine);
                case ExitProgram -> System.exit(0);
            }
        }
    }
}
