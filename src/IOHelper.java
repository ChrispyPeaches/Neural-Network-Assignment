import jdk.jfr.Description;
import jdk.jfr.Name;

import java.io.*;
import java.lang.reflect.Field;
import java.security.InvalidParameterException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class IOHelper {

    //region Output

    /**
     * Set System.out to redirect into an output file
     *
     * @throws IOException If there's an issue creating the output file
     */
    public static PrintStream SetOutputToFileAndReturnOldStream() throws IOException {
        PrintStream prevSystemOut = System.out;

        PrintStream file = new PrintStream("weights-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd|hh:mm:ssa")) + ".txt");
        System.setOut(file);

        return prevSystemOut;
    }

    public static void PrintWeightMatrix(ArrayList<float[][]> weightMatrices, int level) {
        for (int rowIndex = 0; rowIndex < NeuralEngine.GetWeightMatrix(weightMatrices, level).length; rowIndex++) {
            PrintVector(NeuralEngine.GetWeightMatrix(weightMatrices, level)[rowIndex]);
        }
    }

    public static void PrintVector(float[] vector) {
        for (float value : vector) {
            System.out.print(value + ", ");
        }
    }

    public static void SaveWeightsToFile(NeuralEngine engine) throws IOException {
        var oldPrintStream = IOHelper.SetOutputToFileAndReturnOldStream();

        engine.PrintAccuracyResults();
        System.out.println();
        System.out.println("Seed: " + engine.Seed);
        System.out.println();
        System.out.println("Weights:");
        for (int levelIndex = 1; levelIndex < engine.LayerSizes.length; levelIndex++) {
            PrintWeightMatrix(engine.CurrentWeightMatrices, levelIndex);
            System.out.println("");
        }
        System.out.println();
        System.out.println("Biases:");
        for (int levelIndex = 1; levelIndex < engine.LayerSizes.length; levelIndex++) {
            PrintVector(NeuralEngine.GetBiasVector(engine.CurrentBiasVectors, levelIndex));
            System.out.println("");
        }

        // Cleanup
        System.out.flush();
        System.out.close();
        System.setOut(oldPrintStream);
    }

    //endregion

    //region Input
    public static void LoadWeightsFromFile(NeuralEngine engine) throws IOException {
        Scanner inputHandler = new Scanner(System.in);
        File file = null;
        BufferedReader reader = null;
        IOHelper.EngineMode selectedMode = null;
        do {
            System.out.println("Specify an input file");
            String fileName = inputHandler.nextLine();
            try {
                reader = new BufferedReader(new FileReader(fileName));

                // Retreive weights
                String textLine = "";
                do {
                    textLine = reader.readLine();
                } while (!Objects.equals(textLine, "Weights:"));

                textLine = reader.readLine();
                for (int levelIndex = 1; levelIndex < engine.LayerSizes.length; levelIndex++) {
                    float[][] weights = new float[engine.LayerSizes[levelIndex]][engine.LayerSizes[levelIndex - 1]];
                    String[] cellValues = textLine.split(",");
                    for (int rowIndex = 0; rowIndex < weights.length; rowIndex++) {
                        for (int columnIndex = 0; columnIndex < weights[0].length; columnIndex++) {
                            weights[rowIndex][columnIndex] =
                                    Float.parseFloat(cellValues[(rowIndex * weights[0].length) + columnIndex]);
                        }
                    }
                    NeuralEngine.SetWeightMatrix(engine.CurrentWeightMatrices, levelIndex, weights);
                }

                // Retrieve biases
                textLine = "";
                do {
                    textLine = reader.readLine();
                } while (!Objects.equals(textLine, "Biases:"));

                textLine = reader.readLine();
                for (int levelIndex = 1; levelIndex < engine.LayerSizes.length; levelIndex++) {
                    float[] biases = new float[engine.LayerSizes[levelIndex]];
                    String[] cellValues = textLine.split(",");
                    for (int columnIndex = 0; columnIndex < biases.length; columnIndex++) {
                            biases[columnIndex] =
                                    Float.parseFloat(cellValues[columnIndex]);
                        }
                    NeuralEngine.SetBiasVector(engine.CurrentBiasVectors, levelIndex, biases);
                }
            }
            catch (FileNotFoundException e) {
                System.out.println("File not found. Try again.");
            }
        } while (reader == null);
    }

    public static void GetInputsFromFile(
            ArrayList<float[]> inputVectorsReference,
            ArrayList<float[]> expectedOutputsReference,
            List<Integer> dataSetIndices) throws IOException {
        File file = null;
        BufferedReader read = null;
        try {
            read = new BufferedReader(new FileReader("./data_files/mnist_train.csv"));

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

    public static IOHelper.EngineMode GetEngineModeFromInput(boolean weightsAndBiasesLoaded) {
        Scanner inputHandler = new Scanner(System.in);
        IOHelper.EngineMode selectedMode = null;
        do {
            PrintInputOptions(weightsAndBiasesLoaded);
            String inputValue = inputHandler.nextLine();
            Integer formattedInputValue = null;
            try {
                formattedInputValue = Integer.parseInt(inputValue);
                selectedMode = IOHelper.EngineMode.getMode(formattedInputValue);
            }
            catch (Exception e) {
                System.out.println("Invalid parameter");
            }
        } while (selectedMode == null);

        return selectedMode;
    }

    public static void PrintInputOptions(boolean weightsAndBiasesLoaded) {
        System.out.println();
        System.out.println("What would you like to do?");
        Class<EngineMode> reflectedClass = IOHelper.EngineMode.class;
        for (Field field: reflectedClass.getFields()) {
            // If the weights & biases aren't loaded, only print options 1 & 2
            if (!weightsAndBiasesLoaded) {
                if (field.getAnnotation(Name.class).value().contains("1") ||
                        field.getAnnotation(Name.class).value().contains("2")) {
                    System.out.println(field.getAnnotation(Name.class).value());
                }
            }
            else {
                System.out.println(field.getAnnotation(Name.class).value());
            }
        }
    }

    public enum EngineMode {
        @Name(value = "[1] Train the network")
        @Description(value = "Iterate through the 60,000 item MNIST training data set.")
        TrainWithRandomWeights,

        @Name(value = "[2] Load a pre-trained network")
        @Description(value = "Load a weight set (previously generated) from a file.")
        LoadPreTrainedNetwork,

        @Name(value = "[3] Display network accuracy on Training data")
        @Description(value = "Iterate over the 60,000 item MNIST training data set exactly once, " +
                "using the current weight set,and output the statistics.")
        TrainingDataAccuracyDisplay,

        @Name(value = "[4] Display network accuracy on Testing data")
        @Description(value = "Iterate over the 10,000 item MNIST testing data set exactly once, " +
                "using the current weight set, and output the statistics")
        TestingDataAccuracyDisplay,

        @Name(value = "[5] Run network on Testing data showing images and labels")
        @Description(value = "Show (while running the network on TESTING data) for each input image, " +
                "a representation of the image itself, its correct classification, the network’s classification, " +
                "and an indication as to whether or not the network’s classification was correct.")
        RunTestingDataAndShowImages,

        @Name(value = "[6] Display the misclassified Testing images")
        @Description(value = "Similar to option [5] except only display the testing " +
                "images that are misclassified by the network")
        DisplayMisclassifiedTestingImages,

        @Name(value = "[7] Save the network state to file")
        @Description(value = "Save the current weight set to a file.")
        SaveNetworkState,

        @Name(value = "[0] Exit")
        @Description(value = "Exit the program")
        ExitProgram;

        public static EngineMode getMode(int inputMode) {
            return switch (inputMode) {
                case 1 -> TrainWithRandomWeights;
                case 2 -> LoadPreTrainedNetwork;
                case 3 -> TrainingDataAccuracyDisplay;
                case 4 -> TestingDataAccuracyDisplay;
                case 5 -> RunTestingDataAndShowImages;
                case 6 -> DisplayMisclassifiedTestingImages;
                case 7 -> SaveNetworkState;
                case 0 -> ExitProgram;
                default -> throw new InvalidParameterException();
            };
        }
    }

    //endregion
}