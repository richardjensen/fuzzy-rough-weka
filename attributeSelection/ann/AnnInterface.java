package weka.attributeSelection.ann;

import weka.core.Instances;

public interface AnnInterface {
    /**
     * Build the ANN index from the given data.
     * @param data The training instances.
     */
    void buildIndex(Instances data);

    /**
     * Query the index for nearest neighbors.
     * @param vector The query vector.
     * @param k The number of neighbors to return. If k <= 0, may return all candidates found.
     * @return An array of local indices of the nearest neighbors.
     */
    int[] query(double[] vector, int k);
}