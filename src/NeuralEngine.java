import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.*;

public class NeuralEngine {
    //region Class & instance variables
    /** L sizes */
    public final int[] LayerSizes;
    /** W matrices for the current mini batch */
    public ArrayList<float[][]> CurrentWeightMatrices;
    /** b vectors for the current mini batch */
    public ArrayList<float[]> CurrentBiasVectors = new ArrayList<>();
    /** Input, a, and y vectors for the current training set */
    private ArrayList<float[]> CurrentActivationVectors = new ArrayList<>();
    /** Used to gather output activations for debugging output */
    private ArrayList<float[]> OutputVectorsForMinibatch = new ArrayList<>();
    /** The order that the dataset will be broken into mini batches */
    private ArrayList<Integer> DataSetIndicesOrder;
    /** The input vectors for the current mini batch */
    private ArrayList<float[]> InputVectors = new ArrayList<>();

    /**
     * Tracks accuracy for a network's final layer neurons' outputs
     */
    private Dictionary<Integer, OutputResult> OutputResults = new Hashtable<>();

    /** Used for tracking the seed that generated a weight & biases set's original values */
    public Long Seed = null;

    /**
     * Tracks accuracy for a network neuron's output
     */
    public static class OutputResult {
        public int correctOutputs = 0;
        public int totalExpectedOutputs = 0;

        @Override
        public String toString() {
            return correctOutputs + "/" + totalExpectedOutputs;
        }
    }

    /**
     * The size for each mini batch
     */
    public int MiniBatchSize;

    /**
     * The number of mini batches a data set is broken into
     */
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
     * Initialize the engine
     * @param layerSizes A list of the sizes for each layer of the network
     * @param miniBatchSize The size for each mini batch
     */
    public NeuralEngine(
            int[] layerSizes,
            int miniBatchSize) {
        LayerSizes = layerSizes;
        MiniBatchSize = miniBatchSize;
        CurrentWeightMatrices = new ArrayList<>(LayerSizes.length - 1);
    }

    /**
     * Used to demonstrate the engine's capabilities
     * @param dataSetType The type of dataset to demonstrate
     * @param outputType The method for demonstrating the engine
     */
    public void DemoEngine(IOHelper.DataSetType dataSetType,
                           IOHelper.OutputType outputType) throws IOException {
        RunEngine(-1, 1, false, dataSetType, outputType);
    }

    /**
     * Used for training the network
     * @param dataSetType The type of dataset to train on
     * @param learningRate The learning rate for the network
     * @param numberOfEpochs The amount of times to run the network through a dataset
     */
    public void TrainEngine(int learningRate,
                            int numberOfEpochs,
                            IOHelper.DataSetType dataSetType) throws IOException {
        RunEngine(learningRate, numberOfEpochs, true, dataSetType, IOHelper.OutputType.Training);
    }

    //region Neural Engine

    /**
     * Train or test the neural network.
     * @param learningRate the learning rate for gradient descent. Ignored if not training
     * @param numberOfEpochs The amount of epochs to perform. Should be 1 if not training
     * @param isTraining If true, train the network, otherwise test the network
     * @param dataSetType The type of dataset to run through
     * @param outputType The method for communicating with the user
     */
    private void RunEngine(int learningRate,
                           int numberOfEpochs,
                           boolean isTraining,
                           IOHelper.DataSetType dataSetType,
                           IOHelper.OutputType outputType) throws IOException {
        // Minibatch & dataset Setup
        DataSetIndicesOrder = MathHelper.RandomizeDatasetIndicesOrder(GetDataSetSize(dataSetType));
        NumberofMiniBatches = DataSetIndicesOrder.size() / MiniBatchSize;
        CheckMinibatchSize();

        // Training Setup
        if (isTraining) {
            Seed = null;
            LearningRate = learningRate;
            GenerateRandomWeightsAndBiases();
        }

        // Gather training/testing data
        IOHelper.GetInputsAndExpectedOutputsFromCsv(
                InputVectors,
                CorrectOutputVectors,
                dataSetType);


        // Epoch loop
        for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
            if (isTraining) {
                System.out.println("Epoch " + (epochIndex + 1) + ":");
            }

            // Setup accuracy tracking
            OutputResults = new Hashtable<>();
            for (int i = 0; i < LayerSizes[LayerSizes.length - 1]; i++) {
                OutputResults.put(i, new OutputResult());
            }

            for (int miniBatchIndex = 0; miniBatchIndex < NumberofMiniBatches; miniBatchIndex++) {
                if (miniBatchIndex != 0) {
                    System.out.print("\r");
                }
                System.out.print("Running minibatch: " + miniBatchIndex);
                // Setup for gradient descent
                if (isTraining) {
                    BiasGradientSumVectors = new ArrayList<>();
                    WeightGradientSumMatrices = new ArrayList<>();

                    // Create data structure for gradient sums
                    for (int level = 1; level < LayerSizes.length; level++) {
                        BiasGradientSumVectors.add(new float[LayerSizes[level]]);
                        WeightGradientSumMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
                    }
                }

                // Process each of the minibatch's training/testing cases
                for (int trainingCaseIndex = 0; trainingCaseIndex < MiniBatchSize; trainingCaseIndex++) {

                    // Setup activation vectors & current gradients
                    CurrentActivationVectors = new ArrayList<>();

                    /** Input, a, and y vectors for the current training set */
                    if (isTraining) {
                        CurrentBiasGradientVectors = new ArrayList<>();
                        CurrentWeightGradientMatrices = new ArrayList<>();
                    }

                    // Create data structure for activation vectors & gradients
                    for (int level = 0; level < LayerSizes.length; level++) {
                        if (level == 0) {
                            CurrentActivationVectors.add(GetInputOrExpectedOutputVector(
                                    InputVectors,
                                    miniBatchIndex,
                                    trainingCaseIndex));
                        } else {
                            CurrentActivationVectors.add(new float[LayerSizes[level]]);
                            if (isTraining) {
                                CurrentBiasGradientVectors.add(new float[LayerSizes[level]]);
                                CurrentWeightGradientMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
                            }
                        }
                    }

                    // Forward Pass
                    for (int level = 1; level < LayerSizes.length; level++) {
                        // Calculate activation vectors
                        CalculateSigmoid(CurrentActivationVectors, level);
                    }

                    if (isTraining) {
                        PerformBackPropagation(miniBatchIndex, trainingCaseIndex);
                    }

                    if (outputType == IOHelper.OutputType.Accuracy || outputType == IOHelper.OutputType.Training) {
                        TrackOutputAccuracy(miniBatchIndex, trainingCaseIndex);
                    }

                    // Display Ascii image if necessary
                    if (outputType == IOHelper.OutputType.AllImages ||
                            outputType == IOHelper.OutputType.MisclassifiedImages) {
                        // Get expected and actual output for the network
                        Scanner inputHandler = new Scanner(System.in);
                        int output = ConvertOneHotVectorToDigit(GetOutputVector(LayerSizes.length - 1));
                        int expectedOutput =
                                ConvertOneHotVectorToDigit(
                                        GetInputOrExpectedOutputVector(
                                                CorrectOutputVectors,
                                                miniBatchIndex,
                                                trainingCaseIndex));

                        // Handle output to user
                        if (outputType == IOHelper.OutputType.AllImages || output != expectedOutput) {
                            System.out.println(
                                    "Test case #" + (miniBatchIndex * MiniBatchSize + trainingCaseIndex) +
                                            ": Correct classification = " + expectedOutput +
                                            " Network Output = " + output +
                                            ((output == expectedOutput) ? " Correct." : " Incorrect."));
                            IOHelper.PrintAsciiCharacter(
                                    GetInputOrExpectedOutputVector(
                                            InputVectors,
                                            miniBatchIndex,
                                            trainingCaseIndex));
                            System.out.println("Enter 1 to continue. All other values return to main menu");
                            String inputValue = inputHandler.nextLine();
                            if (!inputValue.equals("1")) {
                                return;
                            }
                        }
                    }
                }

                if (isTraining) {
                    PerformGradientDescent(MiniBatchSize);
                }
            }

            PrintAccuracyResults();
        }
    }

    //endregion

    //region Neural network engine helpers

    /**
     * Calculate the z value for each neuron in the target level
     * @param level The target level
     */
    private void CalculateZVector(ArrayList<float[]> CurrentActivationVectors, int level) {
        // For each neuron
        for (int neuronIndex = 0; neuronIndex < GetWeightMatrix(CurrentWeightMatrices, level).length; neuronIndex++) {
            // W * X
            GetOutputVector(level)[neuronIndex] =
                    MathHelper.calculateDotProduct(
                            GetWeightRowVector(CurrentWeightMatrices, level, neuronIndex),
                            GetActivationVector(CurrentActivationVectors, level - 1));
            // + b
            GetOutputVector(level)[neuronIndex] += GetBiasVector(CurrentBiasVectors, level)[neuronIndex];
        }
    }

    /**
     * Use each neuron's z value to calculate the sigmoid value for each neuron in the target level
     * @param level The target level
     */
    private void CalculateSigmoid(ArrayList<float[]> CurrentActivationVectors, int level) {
        CalculateZVector(CurrentActivationVectors, level);

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
    private void CalculateOutputError(int miniBatchIndex, int trainingCaseIndex, int level) {
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
                        (GetOutputVector(level)[neuronIndex]
                                - GetInputOrExpectedOutputVector(
                                        CorrectOutputVectors,
                                        miniBatchIndex,
                                        trainingCaseIndex)[neuronIndex])
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
     * Calculate error in current training case & contribute it to the gradient sums
     * @param trainingCaseIndex The current training case used to retrieve the expected outputs
     */
    private void PerformBackPropagation(int miniBatchIndex, int trainingCaseIndex) {
        for (int level = LayerSizes.length - 1; level > 0; level--) {
            CalculateOutputError(miniBatchIndex, trainingCaseIndex, level);

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

    /**
     * Get the size of the data set for the given data set type
     */
    public static int GetDataSetSize(IOHelper.DataSetType dataSetType) {
        int dataSetSize = 0;
        switch (dataSetType) {
            case Testing -> dataSetSize = 10000;
            case Training -> dataSetSize = 60000;
        }

        return dataSetSize;
    }

    /**
     * Track the output accuracy for each neuron in the final layer
     */
    public void TrackOutputAccuracy(int miniBatchIndex, int trainingCaseIndex) {
            int output = ConvertOneHotVectorToDigit(GetOutputVector(LayerSizes.length - 1));
            int expectedOutput = ConvertOneHotVectorToDigit(
                                        GetInputOrExpectedOutputVector(
                                                CorrectOutputVectors,
                                                miniBatchIndex,
                                                trainingCaseIndex));
            OutputResult a = OutputResults.get(expectedOutput);
            if (output == expectedOutput) {
                a.correctOutputs += 1;
            }
            a.totalExpectedOutputs += 1;
    }

    /**
     * Output the network's accuracy for each of the final layer's nodes and overall
     */
    public void PrintAccuracyResults() {
        int totalCorrect = 0;
        for (int neuronIndex = 0; neuronIndex < LayerSizes[LayerSizes.length - 1]; neuronIndex++) {
            System.out.print(neuronIndex + " = " + OutputResults.get(neuronIndex) + ",\t");
            totalCorrect += OutputResults.get(neuronIndex).correctOutputs;
        }
        System.out.println();
        System.out.println("Accuracy = " + totalCorrect + "/" + (MiniBatchSize * NumberofMiniBatches));
    }

    /**
     * Verify minibatches are a valid size
     */
    public void CheckMinibatchSize() {
        if (DataSetIndicesOrder.size() % MiniBatchSize != 0) {
            throw new IllegalArgumentException();
        }
    }

    /**
     * Using the current seed, or a newly generated one,
     * generate weights and biases for the network
     */
    public void GenerateRandomWeightsAndBiases(){
        // Variable initialization & seed handling
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
     * <p>Note: If miniBatchIndex & trainingCaseIndex aren't available, pass -1 for them to be ignored</p>
     * @param level The target level
     * @return The input vector at the target level
     */
    private float[] GetActivationVector(ArrayList<float[]> activationVectors, int level) {
            return activationVectors.get(level);
    }

    private float[] GetInputOrExpectedOutputVector(ArrayList<float[]> inputOrExpectedOutputVectors,
                                                   int miniBatchIndex,
                                                   int trainingCaseIndex) {
            return inputOrExpectedOutputVectors
                    .get(DataSetIndicesOrder.get((miniBatchIndex * MiniBatchSize) + trainingCaseIndex));
    }

    /**
     * Get the current output vector
     * @param level The target level
     * @return The output vector at the target level
     */
    private float[] GetOutputVector(int level) {
        return CurrentActivationVectors.get(level);
    }

    /**
     * Get the weight matrix for a specified level from the given list of weightMatrices
     * @param weightMatrices A list of weight matrices to retrieve the matrix from
     * @param level The target level of the network
     * @return The target vector
     */
    public static float[][] GetWeightMatrix(ArrayList<float[][]> weightMatrices, int level) {
        return weightMatrices.get(level - 1);
    }

    /**
     * Insert a given matrix into an existing list of matrices
     * @param weightMatrices The list of bias vectors to apply the change to
     * @param level The target level of the network
     * @param newMatrix The target matrix
     */
    public static void SetWeightMatrix(ArrayList<float[][]> weightMatrices, int level, float[][] newMatrix) {
        while (weightMatrices.size() < level) {
            weightMatrices.add(null);
        }
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
        while (biasVectors.size() < level) {
            biasVectors.add(null);
        }
        biasVectors.set(level - 1, newVector);
    }

    //endregion

}
