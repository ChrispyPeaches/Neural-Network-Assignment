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

        PrintStream file = new PrintStream("weights-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd__hh-mm-ssa")) + ".txt");
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

    /**
     * Keep track of what to show the user as the network runs
     */
    public enum OutputType {
        Accuracy,
        AllImages,
        MisclassifiedImages,
        Training
    }

    //endregion

    //region Input

    /**
     * Prompt the user to give a filename and loads weights and biases from that file into the engine
     * <p>Note: Largely uses csv formatting</p>
     * @param engine The neural engine to load the weights and biases into
     */
    public static void LoadWeightsAndBiasesFromFile(NeuralEngine engine) throws IOException {
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

                for (int levelIndex = 1; levelIndex < engine.LayerSizes.length; levelIndex++) {
                    textLine = reader.readLine();
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

                for (int levelIndex = 1; levelIndex < engine.LayerSizes.length; levelIndex++) {
                    textLine = reader.readLine();
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

    /**
     *
     * @param inputVectorsReference
     * @param expectedOutputsReference
     * @param dataSetIndices
     * @param dataSetToRetrieve
     * @throws IOException
     */
    public static void GetInputsFromFile(
            ArrayList<float[]> inputVectorsReference,
            ArrayList<float[]> expectedOutputsReference,
            List<Integer> dataSetIndices,
            DataSetType dataSetToRetrieve) throws IOException {
        File file = null;
        BufferedReader read = null;
        try {
                read = new BufferedReader(new FileReader( dataSetToRetrieve == DataSetType.Training ?
                        "./data_files/mnist_train.csv"
                        : "./data_files/mnist_test.csv"));
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

    /**
     * <ol>
     *     <li>Display the options for using the network</li>
     *     <li>Prompt the user to choose one</li>
     *     <li>Set the engine to be in that 'mode'</li>
     * </ol>
     * @param weightsAndBiasesLoaded Determines which options to show
     * @return The selected engine 'mode'
     */
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

    /**
     * Note: Used gradient from <a href="https://paulbourke.net/dataformats/asciiart/">Paul Bourke's website</a>
     * @param grayscaleValue The value to convert, which should be on a scale from 0 -> 1
     * @return  An ascii character representing the density of the grayscale pixel
     */
    public static char GrayscaleValueToAsciiChar(float grayscaleValue) {
        String asiiDensityGradient = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
        return asiiDensityGradient.charAt((int) Math.floor(grayscaleValue * (asiiDensityGradient.length() - 1)));
    }

    /**
     * Prints an ascii representation for a vector of grayscale pixel values
     * <p>Note: Assumes inputs are a vector for a 1x1 ratio image</p>
     * @param inputVector The vector of grayscale pixel values
     */
    public static void PrintAsciiCharacter(float[] inputVector) {
        int imageWidth = (int) Math.floor(Math.sqrt(inputVector.length));
        for (int rowIndex = 0; rowIndex < inputVector.length; rowIndex++) {
            // Factor in width of image
            if (rowIndex % imageWidth == 0) {
                System.out.println();
            }

            float grayscaleValue = inputVector[rowIndex];
            System.out.print(GrayscaleValueToAsciiChar(grayscaleValue));
        }
        System.out.println();
    }

    /**
     * Display the options for using the network
     * (only display a few options if weights and biases aren't loaded)
     * @param weightsAndBiasesLoaded Determines which options to show
     */
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

    /**
     * The current state of the engine.
     * This is used for minor behavior changes depending on the engine's state
     */
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

    /**
     * The type of data set to use when running the network
     */
    public enum DataSetType {
        Training,
        Testing
    }

    //endregion
}