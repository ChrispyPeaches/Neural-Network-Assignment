import javax.naming.OperationNotSupportedException;
import java.util.ArrayList;

public class NeuralNetwork {
    /** L sizes */
    private final int[] LayerSizes;
    /** eta */
    private int LearningRate;
    private int MiniBatchSize;
    private final int DataSetSize;
    /** W matrices */
    private ArrayList<float[][]> CurrentWeightMatrices = new ArrayList<>();
    /** b vectors */
    private ArrayList<float[]> CurrentBiasVectors = new ArrayList<>();
    private ArrayList<float[][]> WeightGradient = new ArrayList<>();
    private ArrayList<float[][]> BiasGradient = new ArrayList<>();
    /**
     * <p>Current input vector = ActivationVectors[level - 1]</p>
     * <p>Current output vector = ActivationVectors[level]</p>
     */
    private ArrayList<float[]> ActivationVectors = new ArrayList<>();

    public NeuralNetwork(
            int[] layerSizes,
            int dataSetSize) {
        LayerSizes = layerSizes;

        // Build weight matrices, bias vectors, and activation vector lists
        for (int level = 0; level < LayerSizes.length; level++) {
            //GenerateRandomWeights(level);
            UsePregeneratedWeights(level);
            if (level == 0) {
                ActivationVectors.add(new float[]{0,1,0,1});
            } else {
                ActivationVectors.add(new float[LayerSizes[level]]);
            }
        }
        // CurrentInputVector = new float[LayerSizes[0]];
        DataSetSize = dataSetSize;
    }

    public void TrainNetwork(
            int learningRate,
            int miniBatchSize) {
        LearningRate = learningRate;
        MiniBatchSize = miniBatchSize;

        for (int level = 1; level < LayerSizes.length; level++) {
            // Calculate sigmoid into ActivationVectors
            sigmoid(level);
            var a = "breakpoint";
        }
    }

    private void UsePregeneratedWeights(int level) {
        if (level == 0) {
            CurrentWeightMatrices.add(new float[][]{
                    new float[]{-0.21f, 0.72f, -0.25f, 1},
                    new float[]{-0.94f, -0.41f, -0.47f, 0.63f},
                    new float[]{0.15f, 0.55f, -0.49f, -0.75f}
            });
            CurrentBiasVectors.add(new float[]{0.1f, -0.36f, -0.31f});
        }
        else if (level == 1) {
            CurrentWeightMatrices.add( new float[][]{
                    new float[]{0.76f, 0.48f, -0.73f},
                    new float[]{0.34f, 0.89f, -0.23f}
            });
            CurrentBiasVectors.add(new float[]{0.16f, -0.46f});
        }
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
     * @throws OperationNotSupportedException If the dot product is not possible
     */
    private void dotProduct(int level, int weightVectorRowNum) throws UnsupportedOperationException {
        // Error checking
        // Validate matrix1 row count is equal to matrix2 column count
        if (GetWeightMatrixByLevel(level)[weightVectorRowNum].length != GetInputVectorByLevel(level).length) {
            throw new UnsupportedOperationException();
        }

        // Calculation
        float dotProduct = 0;
        for (int index = 0; index < GetInputVectorByLevel(level).length; index++) {
            dotProduct += GetWeightMatrixByLevel(level)[weightVectorRowNum][index] * GetInputVectorByLevel(level)[index];
        }

        // Insert into output vector
        GetOutputVectorByLevel(level)[weightVectorRowNum] = dotProduct;
    }


    /**
     *
     * <p>Stores the result in ActivationVectors </p>
     * @param level
     */
    private void calculateZVector(int level) {
        for (int index = 0; index < GetWeightMatrixByLevel(level).length; index++) {
            dotProduct(level, index);
            GetOutputVectorByLevel(level)[index] += GetBiasVectorByLevel(level)[index];
        }
    }

    private void sigmoid(int level) {
        calculateZVector(level);

        for (int index = 0; index < GetOutputVectorByLevel(level).length; index++) {
            GetOutputVectorByLevel(level)[index] = (float) (1 / (1 + Math.exp(-GetOutputVectorByLevel(level)[index])));
        }
    }

    /*
     * Accessors - passing by reference since they're returning arrays
     */

    private float[] GetInputVectorByLevel(int level) {
        return ActivationVectors.get(level - 1);
    }

    private float[] GetOutputVectorByLevel(int level) {
        return ActivationVectors.get(level);
    }

    private float[][] GetWeightMatrixByLevel(int level) {
        return CurrentWeightMatrices.get(level - 1);
    }

    private float[] GetBiasVectorByLevel(int level) {
        return CurrentBiasVectors.get(level - 1);
    }
}
