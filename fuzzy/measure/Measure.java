package weka.fuzzy.measure;

import java.io.Serializable;
import java.util.BitSet;
import java.util.Map; // --- NEW ---
import java.util.Vector;

import weka.core.Instances;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;

public abstract class Measure implements Serializable {

	static final long serialVersionUID = 1063626753411807303L;
	
	// --- NEW: Add a member to hold the pre-computed neighbor cache ---
	/**
	 * Optional cache mapping an instance index to its candidate neighbors.
	 * If not null, calculations should use this instead of iterating the full dataset.
	 */
	protected Map<Integer, int[]> m_neighborCache = null;
	
	public abstract double calculate(BitSet subset);
	
	public abstract void set(Similarity condSim, Similarity decSim, TNorm tnorm, TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs, int classIndex, Instances ins);
	
	public abstract String globalInfo();

	public abstract String toString();
	
	// --- NEW: Method to inject the neighbor cache from the evaluator ---
	/**
	 * Sets the neighbor cache to be used for optimized calculations.
	 * @param cache A map where the key is an instance index and the value is an array of neighbor indices.
	 */
	public void setNeighborCache(Map<Integer, int[]> cache) {
		this.m_neighborCache = cache;
	}
	
	public String[] getOptions() {
		Vector<String>	result;
	    
	    result = new Vector<String>();
	    return result.toArray(new String[result.size()]);
	}
	
	public void setOptions(String[] options) throws Exception {
		// no options
	}
}