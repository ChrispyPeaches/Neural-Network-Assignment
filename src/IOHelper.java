import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
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
        PrintStream file = new PrintStream("output-" + LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + ".txt");
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
            ArrayList<int[]> expectedOutputsReference,
            List<Integer> dataSetIndices) {
        File file = null;
        Scanner reader = null;
        try {
            file = new File("./data_files/mnist_train.csv");
            reader = new Scanner(file);

        } catch (FileNotFoundException e) {
            System.out.println("File not found");
            throw new RuntimeException(e);
        }


        // Sort the indices to avoid resetting the file reading stream
        Collections.sort(dataSetIndices);

        int currentFileLine = 1;
        for (Integer dataSetIndex : dataSetIndices) {
            int dataSetIndexLineNumber = dataSetIndex + 1;
            String textLine = "";
            // Grab the value on the desired line
            for (int j = currentFileLine; j <= dataSetIndexLineNumber; j++) {
                textLine = reader.nextLine();
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

}