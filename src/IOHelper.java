import jdk.jfr.Description;
import jdk.jfr.Name;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class IOHelper {
    /**
     * Set System.out to redirect into an output file
     *
     * @throws IOException If there's an issue creating the output file
     */
    public static void SetOutputToFile() throws IOException {
        PrintStream file = new PrintStream("output-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd-hh-mm-ss")) + ".txt");
        System.setOut(file);
    }

    public static void PrintWeightMatrix(ArrayList<float[][]> weightMatrices, int level) {
        for (int rowIndex = 0; rowIndex < NeuralEngine.GetWeightMatrix(weightMatrices, level).length; rowIndex++) {
            PrintVector(NeuralEngine.GetWeightMatrix(weightMatrices, level)[rowIndex]);
        }
    }

    public static void PrintVector(float[] biasVector) {
        for (float bias : biasVector) {
            System.out.print(bias + ", ");
        }
        System.out.print("\n");
    }

    public static void ParseCsv(
            ArrayList<float[]> inputVectorsReference,
            ArrayList<float[]> expectedOutputsReference,
            List<Integer> dataSetIndices) throws IOException {
        File file = null;
        BufferedReader read = null;
        try {
            read = new BufferedReader(new FileReader("./data_files/mnist_train.csv"), 8192 * 1000);

        } catch (FileNotFoundException e) {
            System.out.println("File not found");
            throw new RuntimeException(e);
        }

        // Sort the indices to avoid resetting the file reading stream
        Collections.sort(dataSetIndices);

        int currentFileLine = 0;
        for (Integer dataSetIndex : dataSetIndices) {
            int dataSetIndexLineNumber = dataSetIndex;
            String textLine = "";
            // Grab the value on the desired line
            for (int j = currentFileLine; j < dataSetIndexLineNumber; j++) {
                textLine = read.readLine();
                currentFileLine += 1;
            }

            // Gather input and expected output values from the line
            String[] cellValues = textLine.split(",");
            expectedOutputsReference.add(
                    NeuralEngine.ConvertDigitToOneHotVector(Integer.parseInt(cellValues[0])));
            float[] inputs = new float[cellValues.length - 1];
            for (int j = 1; j < cellValues.length; j++) {
                inputs[j - 1] = Float.parseFloat(cellValues[j]) / 255;
            }
            inputVectorsReference.add(inputs);
        }
    }

    public enum EngineMode {
        @Name(value = "Train the network")
        @Description(value = "Iterate through the 60,000 item MNIST training data set.")
        TrainWithRandomWeights,

        @Name(value = "Load a pre-trained network")
        @Description(value = "Load a weight set (previously generated) from a file.")
        LoadPreTrainedNetwork,

        @Name(value = "Display network accuracy on Training data")
        @Description(value = "Iterate over the 60,000 item MNIST training data set exactly once, " +
                "using the current weight set,and output the statistics.")
        TrainingDataAccuracyDisplay,

        @Name(value = "Display network accuracy on Testing data")
        @Description(value = "Iterate over the 10,000 item MNIST testing data set exactly once, " +
                "using the current weight set, and output the statistics")
        TestingDataAccuracyDisplay,

        @Name(value = "Run network on Testing data showing images and labels")
        @Description(value = "Show (while running the network on TESTING data) for each input image, " +
                "a representation of the image itself, its correct classification, the network’s classification, " +
                "and an indication as to whether or not the network’s classification was correct.")
        RunTestingDataAndShowImages,

        @Name(value = "Display the misclassified Testing images")
        @Description(value = "Similar to option [5] except only display the testing " +
                "images that are misclassified by the network")
        DisplayMisclassifiedTestingImages,

        @Name(value = "Save the network state to file")
        @Description(value = "Save the current weight set to a file.")
        SaveNetworkState,

        @Name(value = "Exit")
        @Description(value = "Exit the program")
        ExitProgram;

        public EngineMode getMode(int inputMode) {
            return switch (inputMode) {
                case 1 -> TrainWithRandomWeights;
                case 2 -> LoadPreTrainedNetwork;
                case 3 -> TrainingDataAccuracyDisplay;
                case 4 -> TestingDataAccuracyDisplay;
                case 5 -> RunTestingDataAndShowImages;
                case 6 -> DisplayMisclassifiedTestingImages;
                case 7 -> SaveNetworkState;
                default -> ExitProgram;
            };
        }
    }
}