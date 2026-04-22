package weka.fuzzy.ivmeasure;

import java.util.BitSet;


public class IVGamma extends IVFuzzyMeasure {
	private static final long serialVersionUID = 1063606253456807903L;
	
	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);

		double lower1,lower2;
		double currLower1,currLower2;
		
		double[] val = new double[2];
		double[] dep = new double[2];

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

					// lower approximations of object x
					for (int y = 0; y < m_numInstances; y++) {
						double condSim0 = current[0].getCell(x, y);
						double condSim1 = current[1].getCell(x, y);
						
						double[] sim = fuzzySimilarity(m_classIndex, d, y);
						
						currLower1 = m_Implicator.calculate(condSim1, sim[0]);
						currLower2 = m_Implicator.calculate(condSim0, sim[1]);

						lower1 = Math.min(currLower1, lower1);
						lower2 = Math.min(currLower2, lower2);
						
						if (lower1 == 0 && lower2 == 0)
							break;
					}

					decVals[d][0] = lower1;
					decVals[d][1] = lower2;
					val[0] = Math.max(lower1, val[0]);
					val[1] = Math.max(lower2, val[1]);
					if (val[0] == 1 && val[1] == 1) break;
				}

			}

			
			dep[0] += val[0];
			dep[1] += val[1];
		}
		dep[0] /= (n_objects_d);
		dep[1] /= (n_objects_d);

		return (dep[0] + dep[1]) / 2;
	}

	
	public String globalInfo() {
		return "Interval-valued Gamma evaluator, using the fuzzy rough lower approximations.\n\n";
	}

	public String toString() {
		return "Interval-valued Gamma";
	}

}
