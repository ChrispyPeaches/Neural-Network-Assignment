import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.*;

public class NeuralEngine {
    //region Class & instance variables
    /** L sizes */
    private final int[] LayerSizes;
    /** W matrices for the current mini batch */
    public ArrayList<float[][]> CurrentWeightMatrices = new ArrayList<>();
    /** b vectors for the current mini batch */
    public ArrayList<float[]> CurrentBiasVectors = new ArrayList<>();
    /** Input, a, and y vectors for the current training set */
    private ArrayList<float[]> CurrentActivationVectors = new ArrayList<>();
    /** Used to gather output activations for debugging output */
    private ArrayList<float[]> OutputVectorsForMinibatch = new ArrayList<>();
    /** The order that the dataset will be broken into mini batches */
    private final ArrayList<Integer> DataSetIndicesOrder;
    /** The input vectors for the current mini batch */
    private ArrayList<float[]> InputVectors = new ArrayList<>();

    private Dictionary<Integer, OutputResult> OutputResults = new Hashtable<>();

    public Long Seed;

    public static class OutputResult {
        public int correctOutputs = 0;
        public int totalExpectedOutputs = 0;

        @Override
        public String toString() {
            return correctOutputs + "/" + totalExpectedOutputs;
        }
    }

    public int MiniBatchSize;
    public int NumberofMiniBatches;

    //region Back Propagation Variables

    /** eta - The rate at which the network will learn*/
    private int LearningRate;
    /** The expected output vectors for the current mini batch */
    private ArrayList<float[]> CorrectOutputVectors = new ArrayList<>();
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
     *
     * @param layerSizes
     * @param dataSetSize
     * @param miniBatchSize The size for each mini batch
     */
    public NeuralEngine(
            int[] layerSizes,
            int dataSetSize,
            int miniBatchSize) {
        LayerSizes = layerSizes;
        DataSetIndicesOrder = MathHelper.RandomizeDatasetIndicesOrder(dataSetSize);
        MiniBatchSize = miniBatchSize;
        NumberofMiniBatches = DataSetIndicesOrder.size() / MiniBatchSize;
        Seed = null;
    }

    /**
     * Train the neural network
     * @param learningRate the learning rate for gradient descent
     * @param numberOfEpochs The amount of epochs to perform
     */
    public void TrainNetwork(int learningRate, int numberOfEpochs) throws IOException {
        CheckMinibatchSize(MiniBatchSize);

        // Initialize variables
        Seed = null;
        LearningRate = learningRate;
        GenerateRandomWeightsAndBiases();

        // Epoch loop
        for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
            System.out.println("Epoch " + (epochIndex + 1) + ":");
            OutputResults = new Hashtable<>();
            for (int i = 0; i < LayerSizes[LayerSizes.length - 1]; i++) {
                OutputResults.put(i, new OutputResult());
            }

            // Minibatch loop
            for (int miniBatchIndex = 0; miniBatchIndex < NumberofMiniBatches; miniBatchIndex++) {
                MiniBatchLoop(miniBatchIndex, true);
            }

            PrintAccuracyResults();
        }
    }

    /**
     * Note: Assumes there are already weights and biases initialized
     */
    public void RunNetwork() throws IOException {
        CheckMinibatchSize(MiniBatchSize);

        OutputResults = new Hashtable<>();
        for (int i = 0; i < LayerSizes[LayerSizes.length - 1]; i++) {
            OutputResults.put(i, new OutputResult());
        }

        // Process mini batch
        for (int miniBatchIndex = 0; miniBatchIndex < NumberofMiniBatches; miniBatchIndex++) {
            MiniBatchLoop(miniBatchIndex, false);
        }
    }

    private void MiniBatchLoop(int miniBatchIndex, boolean enableGradientDescent) throws IOException {
        // Reset necessary variables
        OutputVectorsForMinibatch = new ArrayList<>();
        InputVectors = new ArrayList<>();
        CorrectOutputVectors = new ArrayList<>();

        if (enableGradientDescent) {
            BiasGradientSumVectors = new ArrayList<>();
            WeightGradientSumMatrices = new ArrayList<>();

            // Create data structure for gradient sums
            for (int level = 1; level < LayerSizes.length; level++) {
                BiasGradientSumVectors.add(new float[LayerSizes[level]]);
                WeightGradientSumMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
            }
        }

        // Gather training/testing data
        int startingDataIndex = (miniBatchIndex) * MiniBatchSize;
        int endingDataIndex = startingDataIndex + MiniBatchSize;
        IOHelper.ParseCsv(InputVectors, CorrectOutputVectors, DataSetIndicesOrder.subList(startingDataIndex,endingDataIndex));

        // Process each of the minibatch's training/testing cases
        for (int trainingCaseIndex = 0; trainingCaseIndex < MiniBatchSize; trainingCaseIndex++) {
            // Reset activation vectors & current gradients
            CurrentActivationVectors = new ArrayList<>();
            if (enableGradientDescent) {
                CurrentBiasGradientVectors = new ArrayList<>();
                CurrentWeightGradientMatrices = new ArrayList<>();
            }

            // Create data structure for activation vectors & gradients
            for (int level = 0; level < LayerSizes.length; level++) {
                if (level == 0) {
                    CurrentActivationVectors.add(InputVectors.get(trainingCaseIndex));
                } else {
                    CurrentActivationVectors.add(new float[LayerSizes[level]]);
                    if (enableGradientDescent) {
                        CurrentBiasGradientVectors.add(new float[LayerSizes[level]]);
                        CurrentWeightGradientMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
                    }
                }
            }

            // Forward Pass
            for (int level = 1; level < LayerSizes.length; level++) {
                // Calculate activation vectors
                CalculateSigmoid(level);
            }

            // Add the activation vector to the collector for logging
            OutputVectorsForMinibatch.add(CurrentActivationVectors.get(LayerSizes.length - 1));

            // Back Propagation
            if (enableGradientDescent) {
                for (int level = LayerSizes.length - 1; level > 0; level--) {
                    CalculateOutputError(trainingCaseIndex, level);

                    // Build gradient sums
                    SetBiasVector(
                            BiasGradientSumVectors,
                            level,
                            MathHelper.AddTwoVectors(
                                    GetBiasVector(BiasGradientSumVectors, level),
                                    GetBiasVector(CurrentBiasGradientVectors, level)));

                    SetWeightMatrix(
                            WeightGradientSumMatrices,
                            level,
                            MathHelper.addTwoMatrices(
                                    GetWeightMatrix(WeightGradientSumMatrices, level),
                                    GetWeightMatrix(CurrentWeightGradientMatrices, level)));

                }
            }
        }

        if (enableGradientDescent) {
            PerformGradientDescent(MiniBatchSize);
        }

        // Track output accuracy
        for (int trainCaseIndex = 0; trainCaseIndex < OutputVectorsForMinibatch.size(); trainCaseIndex++) {
            int output = ConvertOneHotVectorToDigit(OutputVectorsForMinibatch.get(trainCaseIndex));
            int expectedOutput = ConvertOneHotVectorToDigit(CorrectOutputVectors.get(trainCaseIndex));
            OutputResult a = OutputResults.get(expectedOutput);
            if (output == expectedOutput) {
                a.correctOutputs += 1;
            }
            a.totalExpectedOutputs += 1;
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
                    MathHelper.calculateDotProduct(
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
                float weightDotErrorForNextLevel = MathHelper.calculateDotProduct(
                        GetWeightColumnVector(CurrentWeightMatrices, level + 1, neuronIndex),
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
                    MathHelper.subtractTwoMatrices(
                            GetWeightMatrix(CurrentWeightMatrices, level),
                            MathHelper.scalarMultiplyMatrix(
                                    GetWeightMatrix(WeightGradientSumMatrices, level),
                                    LearningRate / miniBatchSize
                            )
                    )
            );

            // bias_new = bias_old - (learning_rate / size_of_mini_batch) * bias_gradient_sum
            SetBiasVector(
                    CurrentBiasVectors,
                    level,
                    MathHelper.SubtractTwoVectors(
                            GetBiasVector(CurrentBiasVectors, level),
                            MathHelper.ScalarMultiplyVector(
                                    GetBiasVector(BiasGradientSumVectors, level),
                                    LearningRate / miniBatchSize
                            )
                    )
            );
        }
    }

    //endregion

    //region Data helpers

    public void PrintAccuracyResults() {
        // Print accuracy results
        int totalCorrect = 0;
        for (int i = 0; i < LayerSizes[LayerSizes.length - 1]; i++) {
            System.out.println(i + " = " + OutputResults.get(i) + "\t");
            totalCorrect += OutputResults.get(i).correctOutputs;
        }
        System.out.println("Accuracy = " + totalCorrect + "/" + (MiniBatchSize * NumberofMiniBatches));
    }

    /**
     * Verify minibatches are a valid size
     */
    public void CheckMinibatchSize(int miniBatchSize) {
        if (DataSetIndicesOrder.size() % miniBatchSize != 0) {
            throw new IllegalArgumentException();
        }
    }

    /**
     * Hard code the input vectors and the expected outputs
     * @param miniBatchIndex Selects which input vector and expected output vectors to use
     */
    private void SetHardCodedTrainingData(int miniBatchIndex) {
        InputVectors = new ArrayList<>();
        CorrectOutputVectors = new ArrayList<>();
        // Set minibatch's input & correct output vectors
        if (miniBatchIndex == 0) {
            InputVectors.add(new float[]{0,1,0,1});
            CorrectOutputVectors.add(new float[] {0, 1});
            InputVectors.add(new float[]{1,0,1,0});
            CorrectOutputVectors.add(new float[] {1, 0});
        } else if (miniBatchIndex == 1) {
            InputVectors.add(new float[]{0,0,1,1});
            CorrectOutputVectors.add(new float[] {0, 1});
            InputVectors.add(new float[]{1,1,0,0});
            CorrectOutputVectors.add(new float[] {1, 0});
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


    public void GenerateRandomWeightsAndBiases(){
        CurrentWeightMatrices = new ArrayList<>();
        CurrentBiasVectors = new ArrayList<>();
        Random r = new Random();
        long seed = r.nextLong();
        Seed = (Seed == null) ? seed : Seed;
        r.setSeed(Seed);

        for (int inputLevel = 0; inputLevel < LayerSizes.length - 1; inputLevel++) {
            // Generate Weights
            float[][] weights = new float[LayerSizes[inputLevel + 1]][LayerSizes[inputLevel]];
            for (var matrixRow : weights) {
                for (int rowIndex = 0; rowIndex < matrixRow.length; rowIndex++) {
                    matrixRow[rowIndex] = r.nextFloat(-0.9999999f, 1);
                }
            }
            CurrentWeightMatrices.add(weights);

            // Generate Biases
            float[] biases = new float[LayerSizes[inputLevel + 1]];
            for (int biasIndex = 0; biasIndex < biases.length; biasIndex++) {
                biases[biasIndex] = r.nextFloat(-0.9999999f, 1);
            }
            CurrentBiasVectors.add(biases);
        }
    }

    public static float[] ConvertDigitToOneHotVector(int digit) {
        if (digit < 0 || digit > 9) {
            throw new InvalidParameterException();
        }

        float[] oneHotVector = new float[10];
        oneHotVector[digit] = 1;

        return oneHotVector;
    }

    public static int ConvertOneHotVectorToDigit(float[] oneHotVector) {
        int largestIndex = 0;
        for (int i = 1; i < oneHotVector.length; i++) {
            if (oneHotVector[i] > oneHotVector[largestIndex]) {
                largestIndex = i;
            }
        }

        return largestIndex;
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

    public static float[][] GetWeightMatrix(ArrayList<float[][]> weightMatrix, int level) {
        return weightMatrix.get(level - 1);
    }

    /**
     * Insert a given matrix into an existing list of matrices
     * @param weightMatrices The list of bias vectors to apply the change to
     * @param level The target level of the network
     * @param newMatrix The target matrix
     */
    public static void SetWeightMatrix(ArrayList<float[][]> weightMatrices, int level, float[][] newMatrix) {
        weightMatrices.set(level - 1, newMatrix);
    }

    /**
     * Get the row vector for a specified neuron and level from the given list of weightMatrices
     * @param weightMatrices A list of weight matrices to retrieve the vector from
     * @param level The target level of the network
     * @param neuron The target neuron to retrieve a vector for
     * @return The target vector
     */
    public static float[] GetWeightRowVector(ArrayList<float[][]> weightMatrices, int level, int neuron) {
        return weightMatrices.get(level - 1)[neuron];
    }

    /**
     * Get the column vector at a given columnIndex for a target level inside a given list of weightMatrices
     * @param weightMatrices A list of weight matrices to retrieve the vector from
     * @param level The target level of the network
     * @param columnIndex The target column vector's index
     * @return The target vector
     */
    public static float[] GetWeightColumnVector(ArrayList<float[][]> weightMatrices, int level, int columnIndex) {
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
    public static float[] GetBiasVector(ArrayList<float[]> biasVectors, int level) {
        return biasVectors.get(level - 1);
    }

    /**
     * Insert a given vector into an existing list of vectors
     * @param biasVectors The list of bias vectors to apply the change to
     * @param level The target level of the network
     * @param newVector The target vector
     */
    public static void SetBiasVector(ArrayList<float[]> biasVectors, int level, float[] newVector) {
        biasVectors.set(level - 1, newVector);
    }

    //endregion

}
