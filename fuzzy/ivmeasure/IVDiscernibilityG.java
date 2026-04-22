package weka.fuzzy.ivmeasure;

import java.util.BitSet;

public class IVDiscernibilityG extends IVFuzzyMeasure {
	private static final long serialVersionUID = 1063606222456807903L;

	public double calculate(BitSet subset) {
		generatePartition(subset);

		double[] difd = new double[2];
		double[] val = new double[2];

		double[] ret = new double[2];
		ret[0] = 0;
		ret[1] = 0;

		double[] valueRA = new double[2];

		for (int i = 0; i < m_numInstances - 1; i++) {
			for (int j = i + 1; j < m_numInstances; j++) {

				// decision attribute values
				difd = fuzzySimilarity(m_classIndex, i, j);

				//valueRA = 0;
				valueRA[0] = current[0].getCell(i,j);
				valueRA[1] = current[1].getCell(i,j);


				val[0] = m_Implicator.calculate(valueRA[1],difd[0]);
				val[1] = m_Implicator.calculate(valueRA[0],difd[1]);

				ret[0] += val[0];
				ret[1] += val[1];
			}
		}

		return (ret[0]+ret[1])/(n_objects_d * n_objects_d - n_objects_d);
	}

	public String globalInfo() {
		return "Interval-valued discernibility function evaluator.\n\n";
	}

	public String toString() {
		return "Interval-valued discernibility function";
	}


}
