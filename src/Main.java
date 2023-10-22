import java.io.IOException;

public class Main {
    /**
     * Program entry point
     * @param args The arguments given at program start
     * @throws IOException Thrown if there's an issue creating the output file
     */
    public static void main(String[] args) throws IOException {
        IOHelper.SetOutputToFile();
        NeuralEngine main = new NeuralEngine(new int[]{4, 3, 2}, 4);
        main.TrainNetwork(10, 2, 6);
    }
}
