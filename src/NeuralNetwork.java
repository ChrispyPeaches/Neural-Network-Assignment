import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Collections;

public class NeuralNetwork {
    /** L sizes */
    private final int[] LayerSizes;
    /** W matrices */
    private ArrayList<float[][]> CurrentWeightMatrices = new ArrayList<>();
    /** b vectors */
    private ArrayList<float[]> CurrentBiasVectors = new ArrayList<>();
    private ArrayList<float[]> ActivationVectors = new ArrayList<>();
    private ArrayList<Integer> DataSetIndicesOrder;
    private ArrayList<float[]> InputVectors = new ArrayList<>();

    /* Back Propogation Variables */
    /** eta */
    private int LearningRate;
    private ArrayList<int[]> CorrectOutputVectors = new ArrayList<>();
    private ArrayList<float[][]> WeightGradientSumMatrices = new ArrayList<>();
    private ArrayList<float[]> BiasGradientSumVectors = new ArrayList<>();
    private ArrayList<float[][]> CurrentWeightGradientMatrices = new ArrayList<>();
    private ArrayList<float[]> CurrentBiasGradientVectors = new ArrayList<>();

    public NeuralNetwork(
            int[] layerSizes,
            int dataSetSize) {
        LayerSizes = layerSizes;

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
        UsePregeneratedWeights();


        // Set minibatch's input & correct output vectors
        InputVectors.add(new float[]{0,1,0,1});
        InputVectors.add(new float[]{1,0,1,0});
        CorrectOutputVectors.add(new int[] {0, 1});
        CorrectOutputVectors.add(new int[] {1, 0});

        // Minibatch loop
        for (int miniBatchIndex = 0; miniBatchIndex < numberOfMiniBatches; miniBatchIndex++) {
            // Reset gradients sums
            BiasGradientSumVectors = new ArrayList<>();
            WeightGradientSumMatrices = new ArrayList<>();

            // Create data structure for gradient sums
            for (int level = 1; level < LayerSizes.length; level++) {
                BiasGradientSumVectors.add(new float[LayerSizes[level]]);
                WeightGradientSumMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
            }

            for (int trainingCaseIndex = 0; trainingCaseIndex < miniBatchSize; trainingCaseIndex++) {
                // Reset activation vectors & current gradients
                ActivationVectors = new ArrayList<>();
                CurrentBiasGradientVectors = new ArrayList<>();
                CurrentWeightGradientMatrices = new ArrayList<>();

                // Create data structure for activation vectors & gradients
                for (int level = 0; level < LayerSizes.length; level++) {
                    if (level == 0) {
                        ActivationVectors.add(InputVectors.get(trainingCaseIndex));
                    } else {
                        ActivationVectors.add(new float[LayerSizes[level]]);
                        CurrentBiasGradientVectors.add(new float[LayerSizes[level]]);
                        CurrentWeightGradientMatrices.add(new float[LayerSizes[level]][LayerSizes[level - 1]]);
                    }
                }

                // Forward Pass
                for (int level = 1; level < LayerSizes.length; level++) {
                    //GenerateRandomWeights(level);

                    // Calculate activation vectors
                    calculateSigmoid(level);
                }

                // Back Propagation
                for (int level = LayerSizes.length - 1; level > 0; level--) {
                    calculateOutputError(trainingCaseIndex, level);

                    // Build gradient sums
                    SetBiasVector(
                            BiasGradientSumVectors,
                            level,
                            addTwoVectors(
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

            // Gradient Descent


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

    /**
     * Adds vectorA & vectorB by deep copying vectorA, then adding vectorB to the resulting array
     * @param vectorA The first vector to add
     * @param vectorB The second vector to add
     * @return vectorA + vectorB
     */
    private float[] addTwoVectors(float[] vectorA, float[] vectorB) {
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

    private void calculateOutputError(int trainingCaseIndex, int level) {
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
     *
     * <p>Stores the result in ActivationVectors </p>
     * @param level
     */
    private void calculateZVector(int level) {
        // For each neuron
        for (int neuronIndex = 0; neuronIndex < GetWeightMatrix(CurrentWeightMatrices, level).length; neuronIndex++) {
            // W * X
            GetOutputVector(level)[neuronIndex] =
                    calculateDotProduct(GetWeightRowVector(CurrentWeightMatrices, level, neuronIndex), GetInputVector(level));
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

    private void SetWeightMatrix(ArrayList<float[][]> weightMatrices, int level, float[][] newMatrix) {
        weightMatrices.set(level - 1, newMatrix);
    }

    private float[] GetWeightRowVector(ArrayList<float[][]> weightMatrices, int level, int neuron) {
        return weightMatrices.get(level - 1)[neuron];
    }

    private float[] GetWeightColumnVector(ArrayList<float[][]> weightMatrices, int level, int columnNumber) {
        var columnVector = new float[GetWeightMatrix(weightMatrices, level).length];
        for (int rowIndex = 0; rowIndex < GetWeightMatrix(weightMatrices, level).length; rowIndex++) {
            columnVector[rowIndex] = GetWeightMatrix(weightMatrices, level)[rowIndex][columnNumber];
        }
        return columnVector;
    }

    private float[] GetBiasVector(ArrayList<float[]> biasVectors, int level) {
        return biasVectors.get(level - 1);
    }

    private void SetBiasVector(ArrayList<float[]> biasVectors, int level, float[] newVector) {
        biasVectors.set(level - 1, newVector);
    }
}
