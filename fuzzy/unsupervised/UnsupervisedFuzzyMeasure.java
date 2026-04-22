package weka.fuzzy.unsupervised;

import weka.core.*;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Relation;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;

import java.io.Serializable;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Vector;


public abstract class UnsupervisedFuzzyMeasure implements OptionHandler, Serializable  {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	// normalising factor
	double c_divisor = -1; // may need to change this

	public Similarity m_Similarity;
	public Similarity m_SimilarityEq = new SimilarityEq();

	public TNorm m_TNorm;
	public TNorm m_composition;
	public Implicator m_Implicator;
	public SNorm m_SNorm;
	Relation current;
	
	int[] indexes;
	double[] vals; 
	int[] decIndexes;
	double[] decVals;
	int m_numInstances;
	int m_numAttribs;
	int m_classIndex;
	Instances m_trainInstances;
	double n_objects_d;
	
	public UnsupervisedFuzzyMeasure() {
		
	}
	
	public void set(Similarity condSim, Similarity decSim, TNorm tnorm, TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs, int classIndex, Instances ins) {
		m_Similarity = condSim;
		
		m_TNorm = tnorm;
		m_composition = compose;
		m_Implicator = impl;
		m_SNorm = snorm;
		m_numInstances = inst;
		n_objects_d = (double)inst;
		m_numAttribs = attrs;
		m_classIndex = classIndex;
		m_trainInstances = ins;
		current = new Relation(m_numInstances);
	}
	
	/**
	 * Returns a string describing this object.
	 * 
	 * @return 		a description of the evaluator suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public abstract String globalInfo();

	public abstract String toString();
	
	public String[] getOptions() {
		Vector<String>	result;
	    
	    result = new Vector<String>();
	    return result.toArray(new String[result.size()]);
	}

	public Enumeration listOptions() {
		Vector result = new Vector();
		return result.elements();
	}

	public void setOptions(String[] options) throws Exception {
		// no options
		
	}


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
	
	public abstract double calculate(BitSet subset, int attr);
	
	
	public final void generatePartition(BitSet reduct) {
		int nextBit = reduct.nextSetBit(0);
		if (nextBit==m_classIndex) nextBit=reduct.nextSetBit(nextBit+1);

		for (int i = 0; i < m_numInstances; i++) {

			current.setCell(i, i, 1);

			for (int j = i + 1; j < m_numInstances; j++) {
				double rel = fuzzySimilarity(nextBit, i, j);

				current.setCell(i, j, rel);
				current.setCell(j, i, rel);

			}
		}


		if (reduct.cardinality() > 1) {
			for (int o = 0; o < m_numInstances; o++) {
				for (int o1 = o + 1; o1 < m_numInstances; o1++) {
					for (int a = reduct.nextSetBit(nextBit + 1); a >= 0; a = reduct.nextSetBit(a + 1)) {
						if (a!=m_classIndex) {
							double rel = m_composition.calculate(fuzzySimilarity(a, o, o1), current.getCell(o, o1));
							current.setCell(o, o1, rel);
							current.setCell(o1, o, rel);
							if (rel == 0)
								break;
						}
					}
				}
			}
		}
	}
	

	public final static double LN2 = 0.69314718055994531;
	
	//calculate the (fuzzy) entropy for an individual feature. This is used for ranking the features in backward search.
	public final double calculateEntropy(int a) {
		double ret=0;
		
		for (int i=0;i<m_numInstances;i++) {
			double sum1=0;

			for (int j=0;j<m_numInstances;j++) {
				sum1+=fuzzySimilarity(a, i, j);
			}	

			ret += log2(sum1/m_numInstances);
		}

		ret = -ret/m_numInstances;
		
		return ret;
	}
	
	//calculates the granularity of attribute a
	public final double calculateGranularity(int a) {
		double ret=0;
		
		for (int i=0;i<m_numInstances;i++) {
			double sum1=0;

			for (int j=0;j<m_numInstances;j++) {
				sum1+=fuzzySimilarity(a, i, j);
			}
			
			ret+=sum1/m_numInstances;
		}
		
		return 1-(ret/m_numInstances);
	}


	public static final double log2(double x) {
		if (x > 0)
			return Math.log(x) / LN2;
		else
			return 0.0;
	}

	public final double fuzzySimilarity(int attr, int x, int y) {
		double ret = 0;			

		//no decision feature, so each object is distinct
		if (attr<0 && attr==m_classIndex) {
			ret=0;
		}
		else {
			double mainVal=m_trainInstances.instance(x) .value(attr);
			double otherVal=m_trainInstances.instance(y) .value(attr);
			
			//if it's a nominal attribute, then use crisp equivalence
			//otherwise use the general similarity measure
			if (Double.isNaN(mainVal)||Double.isNaN(otherVal)) ret=1;	
			else if (m_trainInstances.attribute(attr).isNominal()) ret = m_SimilarityEq.similarity(attr, mainVal, otherVal);
			else ret = m_Similarity.similarity(attr, mainVal, otherVal);

		}
		return ret;
	}
}

