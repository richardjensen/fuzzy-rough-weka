package weka.fuzzy.ivmeasure;

import java.util.BitSet;


public class IVWeakGamma extends IVFuzzyMeasure {
	private static final long serialVersionUID = 1063606253456807903L;


	public double calculate(BitSet subset) {
		generatePartition(subset);

		double lower1,lower2;
		double currLower1,currLower2;

		double[] dep = new double[2];

		for (int x = 0; x < m_numInstances; x++) {
			int d = x;
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

			dep[0] += lower1;
			dep[1] += lower2;
		}
		dep[0] /= (n_objects_d);
		dep[1] /= (n_objects_d);

		return (dep[0] + dep[1]) / 2;
	}


	public String globalInfo() {
		return "Interval-valued weak Gamma evaluator, using fuzzy rough lower approximations.\n\n";
	}

	public String toString() {
		return "Interval-valued weak Gamma";
	}

}
