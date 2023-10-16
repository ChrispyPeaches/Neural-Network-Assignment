import java.io.IOException;
import java.io.PrintStream;
import java.security.InvalidParameterException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;

public class Main {
    //region Class & instance variables
    /** L sizes */
    private final int[] LayerSizes;
    /** W matrices for the current mini batch */
    private final ArrayList<float[][]> CurrentWeightMatrices = new ArrayList<>();
    /** b vectors for the current mini batch */
    private final ArrayList<float[]> CurrentBiasVectors = new ArrayList<>();
    /** Input, a, and y vectors for the current training set */
    private ArrayList<float[]> CurrentActivationVectors = new ArrayList<>();
    /** Used to gather output activations for debugging output */
    private ArrayList<ArrayList<float[]>> ActivationVectorsForMinibatch = new ArrayList<>();
    /** The order that the dataset will be broken into mini batches */
    private final ArrayList<Integer> DataSetIndicesOrder;
    /** The input vectors for the current mini batch */
    private ArrayList<float[]> InputVectors = new ArrayList<>();

    //region Back Propagation Variables

    /** eta - The rate at which the network will learn*/
    private int LearningRate;
    /** The expected output vectors for the current mini batch */
    private ArrayList<int[]> CorrectOutputVectors = new ArrayList<>();
    /** The weight gradient sum vectors for the current mini batch */
    private ArrayList<float[][]> WeightGradientSumMatrices = new ArrayList<>();
    /** The bias gradient sum vectors for the current mini batch */
    private ArrayList<float[]> BiasGradientSumVectors = new ArrayList<>();
    /** The weight gradient matrices for the current training set */
    private ArrayList<float[][]> CurrentWeightGradientMatrices = new ArrayList<>();
    /** The bias gradient matrices for the current training set */
    private ArrayList<float[]> CurrentBiasGradientVectors = new ArrayList<>();

    //endregion

    //endregion

    /**
     * Program entry point
     * @param args The arguments given at program start
     * @throws IOException Thrown if there's an issue creating the output file
     */
    public static void main(String[] args) throws IOException {
        SetOutputToFile();
        Main main = new Main(new int[]{4, 3, 2}, 4);
        main.TrainNetwork(10, 2, 6);
    }

    public Main(
            int[] layerSizes,
            int dataSetSize) {
        LayerSizes = layerSizes;
        DataSetIndicesOrder = RandomizeDatasetIndicesOrder(dataSetSize);
    }

    /**
     * Set System.out to redirect into an output file
     * @throws IOException If there's an issue creating the output file
     */
    public static void SetOutputToFile() throws IOException {
        PrintStream file = new PrintStream("output-" + LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + ".txt");
        System.setOut(file);
    }

    /**
     * Train the neural network
     * @param learningRate the learning rate for gradient descent
     * @param miniBatchSize The size for each mini batch
     * @param numberOfEpochs The amount of epochs to perform
     */
    public void TrainNetwork(int learningRate, int miniBatchSize, int numberOfEpochs) {
        // Verify minibatches are a valid size
        if (DataSetIndicesOrder.size() % miniBatchSize != 0) {
            throw new IllegalArgumentException();
        }

        // Initialize variables
        LearningRate = learningRate;
        int numberOfMiniBatches = DataSetIndicesOrder.size() / miniBatchSize;
        UsePregeneratedWeightsAndBiases();

        // Epoch loop
        for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
            System.out.println("Epoch " + (epochIndex + 1) + ":");
            System.out.println("=======================");
            // Minibatch loop
            for (int miniBatchIndex = 0; miniBatchIndex < numberOfMiniBatches; miniBatchIndex++) {
                System.out.println("Minibatch " + (miniBatchIndex + 1) + ":");
                // Reset necessary variables
                BiasGradientSumVectors = new ArrayList<>();
                WeightGradientSumMatrices = new ArrayList<>();
                ActivationVectorsForMinibatch = new ArrayList<>();

                // Create data structure for gradient sums
                for (int level = 1; level < LayerSizes.length; level++) {
                    BiasGradientSumVectors.add(new float[LayerSizes[level]]);
                    WeightGradientSumMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
                }

                // Gather training data
                SetTrainingData(miniBatchIndex);

                // Training case loop
                for (int trainingCaseIndex = 0; trainingCaseIndex < miniBatchSize; trainingCaseIndex++) {
                    // Reset activation vectors & current gradients
                    CurrentActivationVectors = new ArrayList<>();
                    CurrentBiasGradientVectors = new ArrayList<>();
                    CurrentWeightGradientMatrices = new ArrayList<>();

                    // Create data structure for activation vectors & gradients
                    for (int level = 0; level < LayerSizes.length; level++) {
                        if (level == 0) {
                            CurrentActivationVectors.add(InputVectors.get(trainingCaseIndex));
                        } else {
                            CurrentActivationVectors.add(new float[LayerSizes[level]]);
                            CurrentBiasGradientVectors.add(new float[LayerSizes[level]]);
                            CurrentWeightGradientMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
                        }
                    }

                    // Forward Pass
                    for (int level = 1; level < LayerSizes.length; level++) {
                        // Calculate activation vectors
                        CalculateSigmoid(level);
                    }

                    // Add the activation vector to the collector for logging
                    ActivationVectorsForMinibatch.add(CurrentActivationVectors);


                    // Back Propagation
                    for (int level = LayerSizes.length - 1; level > 0; level--) {
                        CalculateOutputError(trainingCaseIndex, level);

                        // Build gradient sums
                        SetBiasVector(
                                BiasGradientSumVectors,
                                level,
                                AddTwoVectors(
                                        GetBiasVector(BiasGradientSumVectors, level),
                                        GetBiasVector(CurrentBiasGradientVectors, level)));

                        SetWeightMatrix(
                                WeightGradientSumMatrices,
                                level,
                                addTwoMatrices(
                                        GetWeightMatrix(WeightGradientSumMatrices, level),
                                        GetWeightMatrix(CurrentWeightGradientMatrices, level)));

                    }
                }

                // Perform Gradient Descent
                PerformGradientDescent(miniBatchSize);

                // Print weights and biases for each mini batch and the activations for each training case
                for (int levelIndex = 1; levelIndex < LayerSizes.length; levelIndex++) {
                    System.out.println("L" + (levelIndex - 1) + " -> " + "L" + levelIndex + " --------------");
                    System.out.print("\n");
                    System.out.println("Resulting Weights:");
                    PrintWeightMatrix(CurrentWeightMatrices, levelIndex);
                    System.out.print("\n");
                    System.out.println("Resulting Biases:");
                    PrintVector(GetBiasVector(CurrentBiasVectors, levelIndex));
                    System.out.print("\n");
                    System.out.println("Resulting Activations:");
                    for (int trainCaseIndex = 0; trainCaseIndex < ActivationVectorsForMinibatch.size(); trainCaseIndex++) {
                        System.out.println("Training Case " + (trainCaseIndex + 1) + ":");
                        PrintVector(ActivationVectorsForMinibatch.get(trainCaseIndex).get(levelIndex));
                        System.out.print("\n");
                    }
                    System.out.println("----------------------");
                    System.out.print("\n");
                }
            }
        }
    }

    //region Neural network engine helpers

    /**
     * Calculate the z value for each neuron in the target level
     * @param level The target level
     */
    private void CalculateZVector(int level) {
        // For each neuron
        for (int neuronIndex = 0; neuronIndex < GetWeightMatrix(CurrentWeightMatrices, level).length; neuronIndex++) {
            // W * X
            GetOutputVector(level)[neuronIndex] =
                    calculateDotProduct(
                            GetWeightRowVector(CurrentWeightMatrices, level, neuronIndex),
                            GetInputVector(level));
            // + b
            GetOutputVector(level)[neuronIndex] += GetBiasVector(CurrentBiasVectors, level)[neuronIndex];
        }
    }

    /**
     * Use each neuron's z value to calculate the sigmoid value for each neuron in the target level
     * @param level The target level
     */
    private void CalculateSigmoid(int level) {
        CalculateZVector(level);

        // Perform the sigmoid function on each value in the vector
        for (int index = 0; index < GetOutputVector(level).length; index++) {
            GetOutputVector(level)[index] = (float) (1 / (1 + Math.exp(-GetOutputVector(level)[index])));
        }
    }

    /**
     * Calculate Weight and Bias gradient output error in relation to expected outputs
     * @param trainingCaseIndex The training case to calculate error for
     * @param level The target level
     */
    private void CalculateOutputError(int trainingCaseIndex, int level) {
        if (level > LayerSizes.length - 1) {
            throw new InvalidParameterException();
        }

        for (int neuronIndex = 0; neuronIndex < GetBiasVector(CurrentBiasGradientVectors, level).length; neuronIndex++) {
            // Calculate Bias gradient
            // Final layer output error calculation
            if (level == LayerSizes.length - 1) {
                // Calculate Bias gradient using back propagation Equation 1
                // where j = neuronIndex, L = level
                // (a - y) * a * (1 - a)
                GetBiasVector(CurrentBiasGradientVectors, level)[neuronIndex] =
                        (GetOutputVector(level)[neuronIndex] - CorrectOutputVectors.get(trainingCaseIndex)[neuronIndex])
                                * GetOutputVector(level)[neuronIndex]
                                * (1 - GetOutputVector(level)[neuronIndex]);
            }
            // Hidden layer output error calculation
            else {
                // Calculate Bias gradient using back propagation Equation 2
                // where k = neuronIndex, l = level
                // W_(nextLevel) * Delta_(nextLevel)
                float weightDotErrorForNextLevel = calculateDotProduct(
                        GetWeightColumnVector(CurrentWeightMatrices,level + 1, neuronIndex),
                        GetBiasVector(CurrentBiasGradientVectors, level + 1));

                //
                // weightDotErrorForNextLevel * a_level * (1 - a_level)
                GetBiasVector(CurrentBiasGradientVectors, level)[neuronIndex] =
                        weightDotErrorForNextLevel
                                * GetOutputVector(level)[neuronIndex]
                                * (1 - GetOutputVector(level)[neuronIndex]);
            }

            // Calculate Weight Gradient
            // a_(lastLevel) * Delta_(level)
            for (int activationIndex = 0; activationIndex < GetOutputVector(level - 1).length; activationIndex++) {
                GetWeightRowVector(CurrentWeightGradientMatrices, level, neuronIndex)[activationIndex] =
                        GetOutputVector(level - 1)[activationIndex]
                                * GetBiasVector(CurrentBiasGradientVectors, level)[neuronIndex];
            }
        }


    }

    /**
     * Calculate new weights & biases using gradient descent
     * @param miniBatchSize The size for each mini batch
     */
    private void PerformGradientDescent(float miniBatchSize) {
        for (int level = 1; level < LayerSizes.length; level++) {
            // weight_new = weight_old - (learning_rate / size_of_mini_batch) * weight_gradient_sum
            SetWeightMatrix(
                    CurrentWeightMatrices,
                    level,
                    subtractTwoMatrices(
                            GetWeightMatrix(CurrentWeightMatrices, level),
                            scalarMultiplyMatrix(
                                    GetWeightMatrix(WeightGradientSumMatrices, level),
                                    LearningRate / miniBatchSize
                            )
                    )
            );

            // bias_new = bias_old - (learning_rate / size_of_mini_batch) * bias_gradient_sum
            SetBiasVector(
                    CurrentBiasVectors,
                    level,
                    SubtractTwoVectors(
                            GetBiasVector(CurrentBiasVectors, level),
                            ScalarMultiplyVector(
                                    GetBiasVector(BiasGradientSumVectors, level),
                                    LearningRate / miniBatchSize
                            )
                    )
            );
        }
    }

    //endregion

    //region IO helpers


    private void PrintWeightMatrix(ArrayList<float[][]> weightMatrices, int level) {
        for (int rowIndex = 0; rowIndex < GetWeightMatrix(weightMatrices, level).length; rowIndex++) {
            PrintVector(GetWeightMatrix(weightMatrices, level)[rowIndex]);
        }
    }

    private void PrintVector(float[] biasVector) {
        for (float bias : biasVector) {
            System.out.print(bias + ", ");
        }
        System.out.print("\n");
    }

    //endregion

    //region Data helpers

    /**
     * Randomize the order in which the dataset will be accessed
     * @param dataSetSize The total size of the data set
     * @return A randomized list of indices for the data set
     */
    private static ArrayList<Integer> RandomizeDatasetIndicesOrder(int dataSetSize) {
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < dataSetSize; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        return indices;
    }

    /**
     * Hard code the input vectors and the expected outputs
     * @param miniBatchIndex Selects which input vector and expected output vectors to use
     */
    private void SetTrainingData(int miniBatchIndex) {
        InputVectors = new ArrayList<>();
        CorrectOutputVectors = new ArrayList<>();
        // Set minibatch's input & correct output vectors
        if (miniBatchIndex == 0) {
            InputVectors.add(new float[]{0,1,0,1});
            CorrectOutputVectors.add(new int[] {0, 1});
            InputVectors.add(new float[]{1,0,1,0});
            CorrectOutputVectors.add(new int[] {1, 0});
        } else if (miniBatchIndex == 1) {
            InputVectors.add(new float[]{0,0,1,1});
            CorrectOutputVectors.add(new int[] {0, 1});
            InputVectors.add(new float[]{1,1,0,0});
            CorrectOutputVectors.add(new int[] {1, 0});
        }
    }

    /**
     * Hard code the initial weights & biases
     */
    private void UsePregeneratedWeightsAndBiases() {
        // Weights & Biases for level 0 -> level 1
        CurrentWeightMatrices.add(new float[][]{
                new float[]{-0.21f, 0.72f, -0.25f, 1},
                new float[]{-0.94f, -0.41f, -0.47f, 0.63f},
                new float[]{0.15f, 0.55f, -0.49f, -0.75f}});
        CurrentBiasVectors.add(new float[]{0.1f, -0.36f, -0.31f});

        // Weights & Biases for level 1 -> level 2
        CurrentWeightMatrices.add( new float[][]{
                new float[]{0.76f, 0.48f, -0.73f},
                new float[]{0.34f, 0.89f, -0.23f}});
        CurrentBiasVectors.add(new float[]{0.16f, -0.46f});
    }

    /**
     * Placeholder to be used for implementing true stochastic gradient descent
     * @param level The target level
     */
    private void GenerateRandomWeights(int level){}

    //endregion

    //region Matrix & vector math helpers

    /**
     * <p>Calculates and returns the dot product of the given vectors</p>
     * <p>Note: Assumes vector1 is 1 x n matrix, & vector2 is n x 1 matrix</p>
     * <p>Stores the result in CurrentActivationVectors </p>
     * @throws UnsupportedOperationException If the dot product is not possible
     */
    private float calculateDotProduct(float[] vector1, float[] vector2) throws UnsupportedOperationException {
        // Error checking
        // Validate vector1 column is equal to vector2 row count
        if (vector1.length != vector2.length) {
            throw new UnsupportedOperationException();
        }

        // Calculation
        float dotProduct = 0;
        for (int index = 0; index < vector2.length; index++) {
            dotProduct += vector1[index] * vector2[index];
        }

        // Insert into output vector
        return dotProduct;
    }

    /**
     * Adds matrixA & matrixB by deep copying matrixA, then adding matrixB to the resulting array
     * @param matrixA The first matrix to add
     * @param matrixB The second matrix to add
     * @return matrixA + matrixB
     */
    private float[][] addTwoMatrices(float[][] matrixA, float[][] matrixB) {
        // Validate matrixA & matrixB are the same size
        if (matrixA.length != matrixB.length ||
                matrixA[0].length != matrixB[0].length) {
            throw new UnsupportedOperationException();
        }
        float[][] resultingMatrix = matrixA.clone();

        for (int j = 0; j < matrixA.length; j++) {
            for (int k = 0; k < matrixA[0].length; k++) {
                resultingMatrix[j][k] += matrixB[j][k];
            }
        }
        return resultingMatrix;
    }

    private float[][] subtractTwoMatrices(float[][] matrixA, float[][] matrixB) {
        // Validate matrixA & matrixB are the same size
        if (matrixA.length != matrixB.length ||
                matrixA[0].length != matrixB[0].length) {
            throw new UnsupportedOperationException();
        }
        float[][] resultingMatrix = matrixA.clone();

        for (int j = 0; j < matrixA.length; j++) {
            for (int k = 0; k < matrixA[0].length; k++) {
                resultingMatrix[j][k] -= matrixB[j][k];
            }
        }
        return resultingMatrix;
    }

    /**
     * Multiply a given vector by a given scalar value
     * @param vector The vector to multiply
     * @param scalarMultiple The scalar value to multiply the vector by
     * @return The resulting vector
     */
    private static float[] ScalarMultiplyVector(float[] vector, float scalarMultiple) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalarMultiple;
        }
        return vector;
    }

    /**
     * Multiply a given matrix by a given scalar value
     * @param matrix The matrix to multiply
     * @param scalarMultiple The scalar value to multiply the matrix by
     * @return The resulting matrix
     */
    private static float[][] scalarMultiplyMatrix(float[][] matrix, float scalarMultiple) {
        for (float[] floats : matrix) {
            ScalarMultiplyVector(floats, scalarMultiple);
        }
        return matrix;
    }

    /**
     * Adds vectorA & vectorB by deep copying vectorA, then adding vectorB to the resulting array
     * @param vectorA The first vector to add
     * @param vectorB The second vector to add
     * @return vectorA + vectorB
     */
    private static float[] AddTwoVectors(float[] vectorA, float[] vectorB) {
        // Validate vectorA & vectorB are the same size
        if (vectorA.length != vectorB.length) {
            throw new UnsupportedOperationException();
        }
        float[] resultingVector = vectorA.clone();

        for (int j = 0; j < vectorA.length; j++) {
            resultingVector[j] += vectorB[j];
        }
        return resultingVector;
    }

    /**
     * Subtracts vectorB from vectorA by deep copying vectorA, then subtracting vectorB from the resulting array
     * @param vectorA The minuend vector
     * @param vectorB The subtrahend vector
     * @return vectorA - vectorB
     */
    private static float[] SubtractTwoVectors(float[] vectorA, float[] vectorB) {
        // Validate VectorA & vectorB are the same size
        if (vectorA.length != vectorB.length) {
            throw new UnsupportedOperationException();
        }
        float[] resultingVector = vectorA.clone();

        for (int j = 0; j < vectorA.length; j++) {
            resultingVector[j] -= vectorB[j];
        }

        return resultingVector;
    }

    //endregion

    //region Accessing & mutating helpers

    /**
     * Get the current input vector
     * @param level The target level
     * @return The input vector at the target level
     */
    private float[] GetInputVector(int level) {
        return CurrentActivationVectors.get(level - 1);
    }

    /**
     * Get the current output vector
     * @param level The target level
     * @return The output vector at the target level
     */
    private float[] GetOutputVector(int level) {
        return CurrentActivationVectors.get(level);
    }

    private static float[][] GetWeightMatrix(ArrayList<float[][]> weightMatrix, int level) {
        return weightMatrix.get(level - 1);
    }

    /**
     * Insert a given matrix into an existing list of matrices
     * @param weightMatrices The list of bias vectors to apply the change to
     * @param level The target level of the network
     * @param newMatrix The target matrix
     */
    private static void SetWeightMatrix(ArrayList<float[][]> weightMatrices, int level, float[][] newMatrix) {
        weightMatrices.set(level - 1, newMatrix);
    }

    /**
     * Get the row vector for a specified neuron and level from the given list of weightMatrices
     * @param weightMatrices A list of weight matrices to retrieve the vector from
     * @param level The target level of the network
     * @param neuron The target neuron to retrieve a vector for
     * @return The target vector
     */
    private static float[] GetWeightRowVector(ArrayList<float[][]> weightMatrices, int level, int neuron) {
        return weightMatrices.get(level - 1)[neuron];
    }

    /**
     * Get the column vector at a given columnIndex for a target level inside a given list of weightMatrices
     * @param weightMatrices A list of weight matrices to retrieve the vector from
     * @param level The target level of the network
     * @param columnIndex The target column vector's index
     * @return The target vector
     */
    private static float[] GetWeightColumnVector(ArrayList<float[][]> weightMatrices, int level, int columnIndex) {
        float[] columnVector = new float[GetWeightMatrix(weightMatrices, level).length];
        for (int rowIndex = 0; rowIndex < GetWeightMatrix(weightMatrices, level).length; rowIndex++) {
            columnVector[rowIndex] = GetWeightMatrix(weightMatrices, level)[rowIndex][columnIndex];
        }
        return columnVector;
    }

    /**
     * Retrieve the vector from the biasVectors at the target level
     * @param biasVectors The list of bias vectors to get the vector from
     * @param level The target level of the network
     * @return The target vector
     */
    private static float[] GetBiasVector(ArrayList<float[]> biasVectors, int level) {
        return biasVectors.get(level - 1);
    }

    /**
     * Insert a given vector into an existing list of vectors
     * @param biasVectors The list of bias vectors to apply the change to
     * @param level The target level of the network
     * @param newVector The target vector
     */
    private static void SetBiasVector(ArrayList<float[]> biasVectors, int level, float[] newVector) {
        biasVectors.set(level - 1, newVector);
    }

    //endregion
}
