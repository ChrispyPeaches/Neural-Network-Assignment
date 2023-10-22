import java.util.ArrayList;
import java.util.Collections;

public class MathHelper {
    /**
     * Randomize the order in which the dataset will be accessed
     *
     * @param dataSetSize The total size of the data set
     * @return A randomized list of indices for the data set
     */
    public static ArrayList<Integer> RandomizeDatasetIndicesOrder(int dataSetSize) {
        ArrayList<Integer> indices = new ArrayList<Integer>();
        for (int i = 0; i < dataSetSize; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        return indices;
    }

    //region Matrix & Vector Operations

    /**
     * <p>Calculates and returns the dot product of the given vectors</p>
     * <p>Note: Assumes vector1 is 1 x n matrix, & vector2 is n x 1 matrix</p>
     * <p>Stores the result in CurrentActivationVectors </p>
     *
     * @throws UnsupportedOperationException If the dot product is not possible
     */
    public static float calculateDotProduct(float[] vector1, float[] vector2) throws UnsupportedOperationException {
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
     *
     * @param matrixA The first matrix to add
     * @param matrixB The second matrix to add
     * @return matrixA + matrixB
     */
    public static float[][] addTwoMatrices(float[][] matrixA, float[][] matrixB) {
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

    public static float[][] subtractTwoMatrices(float[][] matrixA, float[][] matrixB) {
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
     *
     * @param vector         The vector to multiply
     * @param scalarMultiple The scalar value to multiply the vector by
     * @return The resulting vector
     */
    public static float[] ScalarMultiplyVector(float[] vector, float scalarMultiple) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalarMultiple;
        }
        return vector;
    }

    /**
     * Multiply a given matrix by a given scalar value
     *
     * @param matrix         The matrix to multiply
     * @param scalarMultiple The scalar value to multiply the matrix by
     * @return The resulting matrix
     */
    public static float[][] scalarMultiplyMatrix(float[][] matrix, float scalarMultiple) {
        for (float[] floats : matrix) {
            ScalarMultiplyVector(floats, scalarMultiple);
        }
        return matrix;
    }

    /**
     * Adds vectorA & vectorB by deep copying vectorA, then adding vectorB to the resulting array
     *
     * @param vectorA The first vector to add
     * @param vectorB The second vector to add
     * @return vectorA + vectorB
     */
    public static float[] AddTwoVectors(float[] vectorA, float[] vectorB) {
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
     *
     * @param vectorA The minuend vector
     * @param vectorB The subtrahend vector
     * @return vectorA - vectorB
     */
    public static float[] SubtractTwoVectors(float[] vectorA, float[] vectorB) {
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
}