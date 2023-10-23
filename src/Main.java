import java.io.IOException;

public class Main {
    /**
     * Program entry point
     * @param args The arguments given at program start
     * @throws IOException Thrown if there's an issue creating the output file
     */
    public static void main(String[] args) throws IOException {
        IOHelper.SetOutputToFile();

        NeuralEngine main = new NeuralEngine(new int[]{784, 15, 10}, 600);
        main.TrainNetwork(3, 10, 1);
    }
}
