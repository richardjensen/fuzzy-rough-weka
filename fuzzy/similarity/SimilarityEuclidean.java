package weka.fuzzy.similarity;

import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Type;

/**
 * Implements a similarity measure based on the normalized Euclidean distance for a single dimension.
 * The similarity is calculated as: 1 - (|v1 - v2| / (max - min)).
 * For nominal attributes, it returns 1 for equality and 0 otherwise.
 */
public class SimilarityEuclidean extends Similarity {

    /** for serialization. */
    private static final long serialVersionUID = -823456890123456789L;

    /**
     * Default constructor.
     */
    public SimilarityEuclidean() {
        super();
    }

    /**
     * Constructor that takes instances.
     * @param data the instances to use.
     */
    public SimilarityEuclidean(Instances data) {
        super(data);
    }
    
    @Override
    public String globalInfo() {
        return "Calculates similarity based on normalized Euclidean distance: S = 1 - (|v1 - v2| / range).";
    }

    @Override
    public String toString() {
        return "Euclidean-based Similarity";
    }

    @Override
    public double similarity(int index, double first, double second) {
        // This check is important. If the range of an attribute is 0,
        // it means all values are the same, so their similarity is maximal (1.0).
        if (attrDifference[index] == 0) {
            return 1.0;
        }

        double diff = Math.abs(first - second);
        double normalized_dist = diff / attrDifference[index];
        
        // Ensure similarity is within [0, 1]
        return Math.max(0.0, 1.0 - normalized_dist);
    }
    
    /**
     * Interval-valued version. For this simple similarity, the interval is a point.
     */
    @Override
    public double[] similarity(int index, double first, double second, double param) {
        double sim = similarity(index, first, second);
        // Returns a point interval [sim, sim]
        return new double[]{sim, sim};
    }
    


	@Override
	public void clean() {
		// TODO Auto-generated method stub
		
	}
}