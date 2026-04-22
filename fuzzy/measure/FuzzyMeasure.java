package weka.fuzzy.measure;

import weka.core.*;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Relation;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;

import java.io.Serializable;
import java.util.BitSet;


public abstract class FuzzyMeasure extends Measure implements  Serializable, RevisionHandler  {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8049778182712265833L;

	// normalising factor
	double c_divisor = -1; // may need to change this

	public Similarity m_Similarity;
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity;

	public TNorm m_TNorm;
	public TNorm m_composition;
	public Implicator m_Implicator;
	public SNorm m_SNorm;
	public Relation current;

	int[] indexes;
	double[] vals; 
	double[] mems;
	int[] decIndexes;
	double[] decVals;
	protected int m_numInstances;
	protected int m_numAttribs;
	protected int m_classIndex;
	protected Instances m_trainInstances;
	public boolean objectMemberships=false;

	public double n_objects_d;

	public FuzzyMeasure() {

	}


	public void set(Similarity condSim, Similarity decSim, TNorm tnorm, TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs, int classIndex, Instances ins) {
		m_Similarity = condSim;
		m_DecisionSimilarity = decSim;
		m_TNorm = tnorm;
		m_composition = compose;
		m_Implicator = impl;
		m_SNorm = snorm;
		m_numInstances = inst;
		n_objects_d = (double)inst;
		m_numAttribs = attrs;
		m_classIndex = classIndex;
		m_trainInstances = ins;
		mems = new double[m_numInstances];
		current = new Relation(m_numInstances);
		m_SimilarityEq.setInstances(ins);

		//prepare optimization information
		decIndexes = new int[m_numInstances];

		for (int i = 0; i < m_numInstances; i++) {
			boolean same = false;

			for (int j = i + 1; j < m_numInstances; j++) {

				for (int d = 0; d < m_numInstances; d++) {
					double decSim1 = fuzzySimilarity(m_classIndex, i, d);
					double decSim2 = fuzzySimilarity(m_classIndex, j, d);

					if (decSim1!=decSim2) {same=false; break;}
					else same=true;

				}
				if (same) {
					decIndexes[j] = i + 1;
					break;
				} // i+1 as default is 0
			}

		}


	}


	/**
	 * Returns a string describing this object.
	 * 
	 * @return 		a description of the evaluator suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public abstract String globalInfo();

	public abstract String toString();



	public final void setIndexes(Relation rel) {
		int m_numInstances = rel.size;

		for (int i = 0; i < m_numInstances; i++) {
			boolean same = false;

			for (int j = i + 1; j < m_numInstances; j++) {

				for (int d = 0; d < m_numInstances; d++) {
					if (rel.getCell(i, d) != rel.getCell(j, d)) {
						same = false;
						break;
					} else
						same = true;
				}
				if (same) {
					indexes[j] = i + 1;
					break;
				} // i+1 as default is 0
			}

		}
	}

	public abstract double calculate(BitSet subset);

	public double granularity(int a) {
		double granularity=0;
		
		for (int x = 0; x < m_numInstances; x++) {
			double sum1=0;

			// lower approximations of object x
			for (int y = 0; y < m_numInstances; y++) {
				sum1 += fuzzySimilarity(a, x, y);
			}

			granularity +=  sum1;
		} 

		//from https://www.sciencedirect.com/science/article/pii/S0020025512003805
		granularity/=(m_numInstances*m_numInstances);

		return granularity;
	}
	
	public double granularity2(int a) {
		double ret=0;
		double denom= m_numInstances-1;

		for (int i=0;i<m_numInstances;i++) {
			double sum1=0;

			for (int j=0;j<m_numInstances;j++) {
				sum1+=fuzzySimilarity(a, i, j);
			}

			ret+=(m_numInstances-sum1)/denom;
		}

		return 1-(ret/m_numInstances);

	}

	public double[] objectMemberships(BitSet subset) {
		objectMemberships=true;
		calculate(subset);
		return mems;
	}

	public double[] objectMemberships() {
		BitSet full = new BitSet(m_numAttribs);
		for (int a = 0; a < m_numAttribs - 1; a++)	full.set(a);
		calculate(full);
		return mems;
	}


	public final double getConsistency() {
		BitSet full = new BitSet(m_numAttribs);
		for (int a = 0; a < m_numAttribs; a++)
			if (a!=m_classIndex) full.set(a);


		c_divisor = calculate(full);

		if (c_divisor == 0 || c_divisor == Double.NaN) {
			System.err.println("\n*** Inconsistent data (full dataset value = "
					+ c_divisor + " for this measure) ***\n");
			c_divisor=1;
			//System.exit(1);
		}

		return c_divisor;
	}

	public final void generatePartition(BitSet reduct) {
	    // --- START OF MODIFICATION ---
	    // This entire method is replaced to be cache-aware.

	    // Initialize the relation matrix. Diagonal elements are always 1 (self-similar).
	    // For the non-cached approach, we will calculate all pairs.
	    // For the cached approach, off-diagonal elements will default to 0 unless specified in the cache.
	    for (int i = 0; i < m_numInstances; i++) {
	        current.setCell(i, i, 1.0);
	        // If using cache, initialize all other similarities to 0.
	        // A similarity of 0 means the pair will not lower the dependency score.
	        if (m_neighborCache != null) {
	            for (int j = i + 1; j < m_numInstances; j++) {
	                current.setCell(i, j, 0.0);
	            }
	        }
	    }

	    if (reduct.cardinality() == 0) {
	        return; // Nothing to do for an empty set.
	    }

	    // Find the first attribute to process.
	    int firstBit = reduct.nextSetBit(0);
	    if (firstBit == m_classIndex) {
	        firstBit = reduct.nextSetBit(firstBit + 1);
	    }
	    // If no valid attributes, we are done.
	    if (firstBit < 0) {
	        return;
	    }

	    // This is the main N^2 loop that we need to optimize.
	    for (int i = 0; i < m_numInstances; i++) {
	        // Determine which instances 'j' to compare against 'i'.
	        int[] neighbors;
	        if (m_neighborCache != null && m_neighborCache.containsKey(i)) {
	            // OPTIMIZED PATH: Use the pre-computed neighbor cache.
	            neighbors = m_neighborCache.get(i);
	        } else {
	            // DEFAULT (NONE) PATH: Generate a sequence from i+1 to N-1 for a full pairwise comparison.
	            neighbors = new int[m_numInstances - (i + 1)];
	            for (int k = 0; k < neighbors.length; k++) {
	                neighbors[k] = i + 1 + k;
	            }
	        }

	        // Now iterate only over the required pairs.
	        for (int j : neighbors) {
	            // Ensure we only compute for the upper triangle of the matrix (j > i).
	            if (j <= i) continue;

	            // Calculate similarity using the first attribute.
	            double composedSimilarity = fuzzySimilarity(firstBit, i, j);

	            // If there are more attributes, compose them using the T-Norm.
	            if (reduct.cardinality() > 1) {
	                for (int a = reduct.nextSetBit(firstBit + 1); a >= 0 && composedSimilarity > 0; a = reduct.nextSetBit(a + 1)) {
	                    if (a != m_classIndex) {
	                        composedSimilarity = m_composition.calculate(composedSimilarity, fuzzySimilarity(a, i, j));
	                    }
	                }
	            }
	            
	            // Set the final composed similarity in the relation matrix.
	            current.setCell(i, j, composedSimilarity);
	        }
	    }
	    // --- END OF MODIFICATION ---
	}
	

	public final double fuzzySimilarity(int attr, int x, int y) {
		double ret = 0;			

		//no decision feature, so each object is distinct
		if (attr<0 && attr==m_classIndex) {
			if (x==y) ret=1;
			else ret=0;
		}
		else {

			//semi-supervised approach (class index exists but some values may be missing)
			/*if (attr==m_classIndex) {
				if (m_trainInstances.instance(x).isMissing(attr) || m_trainInstances.instance(y).isMissing(attr)) {
					if (x==y) return 1;
					else return 0;
				}
			}*/
			
			//more general approach - missing values and/or semi-supervised
			if (m_trainInstances.instance(x).isMissing(attr) || m_trainInstances.instance(y).isMissing(attr)) {
				if (attr==m_classIndex) {//semi-supervised
					if (x==y) return 1;
					else return 0;
				}
				else { //missing conditional values
					return 1; //1 is returned as in the worst case the two values are fully similar
				}
			}
			
			
			//normal approach
			double mainVal=m_trainInstances.instance(x).value(attr);
			double otherVal=m_trainInstances.instance(y).value(attr);

			//if it's the class attribute, use the class similarity measure
			//if it's a nominal attribute, then use crisp equivalence
			//otherwise use the general similarity measure
			if (Double.isNaN(mainVal)||Double.isNaN(otherVal)) ret=1;	
			else if (attr==m_classIndex) ret = m_DecisionSimilarity.similarity(attr, mainVal, otherVal);
			else if (m_trainInstances.attribute(attr).isNumeric()) ret = m_Similarity.similarity(attr, mainVal, otherVal);
			else ret = m_SimilarityEq.similarity(attr, mainVal, otherVal);

		}
		return ret;
	}

	
}
