package weka.attributeSelection.ann;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.fuzzy.similarity.Similarity;

/**
 * Utility class for Approximate Nearest Neighbor algorithms.
 * Includes data conversion and the hardcoded Euclidean distance function.
 */
public final class AnnUtils {

    @FunctionalInterface
    public interface DistanceFunction {
        double compute(double[] v1, double[] v2);
    }

    public static class QueueItem {
        public final int index;
        public final double priority;

        public QueueItem(int index, double priority) {
            this.index = index;
            this.priority = priority;
        }
    }

    public static double[][] instancesToMatrix(Instances data) {
        int numInstances = data.numInstances();
        int numFeatures = data.numAttributes() - (data.classIndex() >= 0 ? 1 : 0);
        double[][] matrix = new double[numInstances][numFeatures];
        for (int i = 0; i < numInstances; i++) {
            matrix[i] = instanceToVector(data.instance(i));
        }
        return matrix;
    }

    public static double[] instanceToVector(Instance instance) {
        int numAttrs = instance.numAttributes();
        int classIndex = instance.classIndex();
        double[] vector = new double[numAttrs - (classIndex >= 0 ? 1 : 0)];
        int vecIndex = 0;
        for (int i = 0; i < numAttrs; i++) {
            if (i == classIndex) continue;
            vector[vecIndex++] = instance.value(i);
        }
        return vector;
    }

    /**
     * Creates a hardcoded Normalized Euclidean Distance function.
     * 
     * This ignores the 'similarity' parameter to ensure the ANN is built
     * in a well-behaved geometric space (Euclidean), which is critical for
     * the performance and stability of algorithms like HNSW and NSW.
     *
     * @param data The dataset schema (used to calculate ranges for normalization).
     * @param similarity Ignored (kept for signature compatibility).
     * @return A DistanceFunction calculating Normalized Euclidean Distance.
     */
    public static DistanceFunction createFrfsDistanceFunction(Instances data, Similarity similarity) {
        int numAttrs = data.numAttributes();
        int classIndex = data.classIndex();
        int numFeatures = numAttrs - (classIndex >= 0 ? 1 : 0);

        // Arrays to store normalization info and type info
        final double[] ranges = new double[numFeatures];
        final boolean[] isNominal = new boolean[numFeatures];

        // Map vector indices (0..F) back to original attribute indices (0..A)
        int[] attrIndexMap = new int[numFeatures];
        int k = 0;
        for (int i = 0; i < numAttrs; i++) {
            if (i != classIndex) {
                attrIndexMap[k++] = i;
            }
        }

        // Pre-compute ranges for numeric attributes to perform normalization
        for (int i = 0; i < numFeatures; i++) {
            int originalIndex = attrIndexMap[i];
            Attribute attr = data.attribute(originalIndex);

            if (attr.isNominal()) {
                isNominal[i] = true;
                ranges[i] = 1.0; // Dummy value, not used for nominals
            } else {
                isNominal[i] = false;
                // Use Weka's cached stats if available, otherwise defaults imply no normalization
                weka.core.AttributeStats stats = data.attributeStats(originalIndex);
                double min = stats.numericStats.min;
                double max = stats.numericStats.max;
                
                if (Double.isNaN(min) || Double.isNaN(max)) {
                    ranges[i] = 1.0; // Fallback if stats unavailable
                } else {
                    ranges[i] = max - min;
                    if (ranges[i] < 1e-9) ranges[i] = 1.0; // Prevent division by zero for constant attributes
                }
            }
        }

        // Return the optimized Lambda
        return (v1, v2) -> {
            double sumSq = 0.0;
            for (int i = 0; i < numFeatures; i++) {
                double diff;
                
                if (isNominal[i]) {
                    // Nominal distance: 0 if same, 1 if different
                    // (Comparison of doubles is safe here as Weka uses integer indices)
                    diff = (v1[i] == v2[i]) ? 0.0 : 1.0;
                } else {
                    // Numeric distance: Normalized difference
                    diff = (v1[i] - v2[i]) / ranges[i];
                }
                
                sumSq += diff * diff;
            }
            return Math.sqrt(sumSq);
        };
    }
}