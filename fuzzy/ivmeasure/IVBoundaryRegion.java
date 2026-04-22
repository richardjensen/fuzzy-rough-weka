package weka.fuzzy.ivmeasure;

import java.util.BitSet;

import weka.core.Instances;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;

public class IVBoundaryRegion extends IVFuzzyMeasure {
	private static final long serialVersionUID = 1063606253456804903L;
	double denom;
	double decCardinality;
	double[] decisionLength;
	
	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);
		
		decisionLength = new double[m_numInstances];

		for (int x = 0; x < m_numInstances; x++) {
			for (int y = 0; y < m_numInstances; y++) {
				double[] decS = fuzzySimilarity(m_classIndex, x, y);
				decisionLength[x] += decS[0];
			}

			decCardinality += 1 / decisionLength[x];
		}
	}
	
	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);

		double lower1,upper1;
		double lower2,upper2;

		double[] val = new double[2];
		val[0] = 0;
		val[1] = 0; // positive region
		double[] dep = new double[2];
		dep[0] = 0;
		dep[1] = 0; // dependency

		double[] sim = new double[2];
		double[][] decVals;
		// for optimization
		indexes = new int[m_numInstances];

		double denom = (n_objects_d * decCardinality);


		for (int x = 0; x < m_numInstances; x++) {
			val[0] = 0;
			val[1] = 0;
			decVals = new double[m_numInstances][2];
			
			for (int d = 0; d < m_numInstances; d++) {
				if (decIndexes[d] != 0) {
					decVals[d][0] = decVals[decIndexes[d] - 1][0];
					decVals[d][1] = decVals[decIndexes[d] - 1][1];
				} else {
					lower1 = 1;
					lower2 = 1;
					upper1 = 0;
					upper2 = 0;

					// lower and upper approximations of object x
					for (int y = 0; y < m_numInstances; y++) {
						double condSim0 = current[0].getCell(x, y);
						double condSim1 = current[1].getCell(x, y);
						
						sim = fuzzySimilarity(m_classIndex, d, y);

						lower1 = Math.min(lower1, m_Implicator.calculate(condSim1, sim[0]));
						lower2 = Math.min(lower2, m_Implicator.calculate(condSim0, sim[1]));
						
						upper1 = Math.max(upper1, m_TNorm.calculate(condSim0, sim[0]));
						upper2 = Math.max(upper2, m_TNorm.calculate(condSim1, sim[1]));
						
						if (lower2==0 && upper1 == 1) break;
					}

					decVals[d][0] = (upper1-lower1) /decisionLength[d];
					decVals[d][1] = (upper2-lower2) /decisionLength[d];
				}
				val[0]+=decVals[d][0];
				val[1]+=decVals[d][1];
			}
			
			dep[0] += val[0];
			dep[1] += val[1];
		}

		return 1-((dep[0]/denom + dep[1]/denom) / 2);
	}

	
	public String globalInfo() {
		return "Interval-valued boundary region-based evaluator, using the fuzzy rough lower and upper approximations.\n\n";
	}

	public String toString() {
		return "Interval-valued boundary measure";
	}

}
