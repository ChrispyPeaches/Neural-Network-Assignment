import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Collections;

public class NeuralNetwork {
    /** L sizes */
    private final int[] LayerSizes;
    /** eta */
    private int LearningRate;
    /** W matrices */
    private ArrayList<float[][]> CurrentWeightMatrices = new ArrayList<>();
    /** b vectors */
    private ArrayList<float[]> CurrentBiasVectors = new ArrayList<>();
    private ArrayList<float[]> ActivationVectors = new ArrayList<>();

    /* Back Propogation Variables */
    private ArrayList<int[]> CorrectOutputVectors = new ArrayList<>();
    private ArrayList<float[][]> WeightGradientSumMatrix = new ArrayList<>();
    private ArrayList<float[]> BiasGradientSumMatrix = new ArrayList<>();
    private ArrayList<float[][]> CurrentWeightGradientMatrix = new ArrayList<>();
    private ArrayList<float[]> CurrentBiasGradientMatrix = new ArrayList<>();
    private ArrayList<Integer> DataSetIndicesOrder;

    public NeuralNetwork(
            int[] layerSizes,
            int dataSetSize) {
        LayerSizes = layerSizes;

        // Build weight matrices, bias vectors, and activation vector lists
        UsePregeneratedWeights();
        for (int level = 0; level < LayerSizes.length; level++) {
            if (level == 0) {
                ActivationVectors.add(new float[]{0,1,0,1});
            } else {
                ActivationVectors.add(new float[LayerSizes[level]]);
            }
        }

        // CurrentInputVector = new float[LayerSizes[0]];
        DataSetIndicesOrder = RandomizeDatasetIndicesOrder(dataSetSize);
    }

    public void TrainNetwork(int learningRate, int miniBatchSize) {
        // Verify minibatch is a valid size
        if (DataSetIndicesOrder.size() % miniBatchSize != 0) {
            throw new IllegalArgumentException();
        }
        // Initialize variables
        LearningRate = learningRate;
        //int numberOfMiniBatches = miniBatchSize / DataSetIndicesOrder.size();
        int numberOfMiniBatches = 1;

        for (int miniBatchIndex = 0; miniBatchIndex < numberOfMiniBatches; miniBatchIndex++) {
            // Set minibatch's correct output vectors
            CorrectOutputVectors.add(new int[] {0, 1});

            // Forward Pass
            for (int level = 1; level < LayerSizes.length; level++) {
                //GenerateRandomWeights(level);

                // Setup for back propagation
                CurrentBiasGradientMatrix.add(new float[LayerSizes[level]]);
                CurrentWeightGradientMatrix.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);

                // Calculate activation vectors
                calculateSigmoid(level);
            }

            // Back Propagation
            for (int level = LayerSizes.length - 1; level > 0; level--) {
                calculateOutputError(miniBatchIndex, level);
            }
        }
    }

    private void UsePregeneratedWeights() {
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

    private void GenerateRandomWeights(int level){
        // CurrentBiasVectors = new float[LayerSizes[LayerSizes.length - 1]];
        // CurrentWeightMatrices = new float[LayerSizes[0]][LayerSizes[1]];
        return;
    }

    /**
     * <p>Calculates and returns the dot product of the given vectors</p>
     * <p>Note: Assumes vector1 is 1 x n matrix, & vector2 is n x 1 matrix</p>
     * <p>Stores the result in ActivationVectors </p>
     * @throws UnsupportedOperationException If the dot product is not possible
     */
    private float dotProduct (float[] vector1, float[] vector2) throws UnsupportedOperationException {
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

    private void calculateOutputError(int miniBatchIndex, int level) {
        if (level > LayerSizes.length - 1) {
            throw new InvalidParameterException();
        }

        for (int neuronIndex = 0; neuronIndex < GetBiasVector(CurrentBiasGradientMatrix, level).length; neuronIndex++) {
            // Calculate Bias gradient
            // Final layer output error calculation
            if (level == LayerSizes.length - 1) {
                // Calculate Bias gradient using back propagation Equation 1
                // where j = neuronIndex, L = level
                // (a - y) * a * (1 - a)
                GetBiasVector(CurrentBiasGradientMatrix, level)[neuronIndex] =
                        (GetOutputVector(level)[neuronIndex] - CorrectOutputVectors.get(miniBatchIndex)[neuronIndex])
                        * GetOutputVector(level)[neuronIndex]
                        * (1 - GetOutputVector(level)[neuronIndex]);
            }
            // Hidden layer output error calculation
            else {
                // Calculate Bias gradient using back propagation Equation 2
                // where k = neuronIndex, l = level
                    // W_(nextLevel) * Delta_(nextLevel)
                    float weightDotErrorForNextLevel = dotProduct(
                            GetWeightColumnVector(CurrentWeightMatrices,level + 1, neuronIndex),
                            GetBiasVector(CurrentBiasGradientMatrix, level + 1));

                    //
                    // weightDotErrorForNextLevel * a_level * (1 - a_level)
                    GetBiasVector(CurrentBiasGradientMatrix, level)[neuronIndex] =
                            weightDotErrorForNextLevel
                            * GetOutputVector(level)[neuronIndex]
                            * (1 - GetOutputVector(level)[neuronIndex]);
                }

            // Calculate Weight Gradient
            // a_(lastLevel) * Delta_(level)
            for (int activationIndex = 0; activationIndex < GetOutputVector(level - 1).length; activationIndex++) {
                GetWeightRowVector(CurrentWeightGradientMatrix, level, neuronIndex)[activationIndex] =
                        GetOutputVector(level - 1)[activationIndex]
                        * GetBiasVector(CurrentBiasGradientMatrix, level)[neuronIndex];
            }
        }


    }

    /**
     *
     * <p>Stores the result in ActivationVectors </p>
     * @param level
     */
    private void calculateZVector(int level) {
        // For each neuron
        for (int neuronIndex = 0; neuronIndex < GetWeightMatrix(CurrentWeightMatrices, level).length; neuronIndex++) {
            // W * X
            GetOutputVector(level)[neuronIndex] =
                    dotProduct(GetWeightRowVector(CurrentWeightMatrices, level, neuronIndex), GetInputVector(level));
            // + b
            GetOutputVector(level)[neuronIndex] += GetBiasVector(CurrentBiasVectors, level)[neuronIndex];
        }
    }

    private void calculateSigmoid(int level) {
        calculateZVector(level);

        // Perform the sigmoid function on each value in the vector
        for (int index = 0; index < GetOutputVector(level).length; index++) {
            GetOutputVector(level)[index] = (float) (1 / (1 + Math.exp(-GetOutputVector(level)[index])));
        }
    }

    private ArrayList<Integer> RandomizeDatasetIndicesOrder(int dataSetSize) {
        var a = new ArrayList<Integer>();
        for (int i = 0; i < dataSetSize; i++) {
            a.add(i);
        }
        Collections.shuffle(a);
        return a;
    }

    private float[] GetInputVector(int level) {
        return ActivationVectors.get(level - 1);
    }

    private float[] GetOutputVector(int level) {
        return ActivationVectors.get(level);
    }

    private float[][] GetWeightMatrix(ArrayList<float[][]> weightMatrix, int level) {
        return weightMatrix.get(level - 1);
    }

    private float[] GetWeightRowVector(ArrayList<float[][]> weightMatrix, int level, int neuron) {
        return weightMatrix.get(level - 1)[neuron];
    }

    private float[] GetWeightColumnVector(ArrayList<float[][]> weightMatrix, int level, int columnNumber) {
        var columnVector = new float[GetWeightMatrix(weightMatrix, level).length];
        for (int rowIndex = 0; rowIndex < GetWeightMatrix(weightMatrix, level).length; rowIndex++) {
            columnVector[rowIndex] = GetWeightMatrix(weightMatrix, level)[rowIndex][columnNumber];
        }
        return columnVector;
    }

    private float[] GetBiasVector(ArrayList<float[]> biasMatrix, int level) {
        return biasMatrix.get(level - 1);
    }
}
