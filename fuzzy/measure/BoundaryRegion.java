package weka.fuzzy.measure;

import java.util.BitSet;

import weka.core.Instances;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;


public class BoundaryRegion extends FuzzyMeasure implements TechnicalInformationHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	private double[] lowers,uppers;
	double denom;
	double decCardinality;
	double[] decisionLength;

	public BoundaryRegion() {
		super();
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);

		decisionLength = new double[m_numInstances];

		for (int x = 0; x < m_numInstances; x++) {
			for (int y = 0; y < m_numInstances; y++) {
				double decS = fuzzySimilarity(m_classIndex, x, y);
				decisionLength[x] += decS;
			}

			decCardinality += 1 / decisionLength[x];
		}
	}

	public double calculate2(BitSet subset) {
		generatePartition(subset);

		double lower;
		lowers = new double[m_numInstances];
		double currLower = 0;
		uppers = new double[m_numInstances];
		double ret = 0;
		double upper = 0;
		double currUpper = 0;
		double denom = 0;
		denom = (n_objects_d * decCardinality);

		for (int x = 0; x < m_numInstances; x++) {
			double temp=0;

			for (int d = 0; d < m_numInstances; d++) {
				lower = 1; upper=0;

				// lower approximations of object x
				for (int y = 0; y < m_numInstances; y++) {
					double condSim = current.getCell(x, y);

					currLower = m_Implicator.calculate(
							condSim,
							fuzzySimilarity(m_classIndex, d, y));

					currUpper = m_TNorm.calculate(
							condSim,
							fuzzySimilarity(m_classIndex, d, y));

					upper = Math.max(currUpper, upper);
					lower = Math.min(currLower, lower);
				}

				lowers[d] = lower;
				uppers[d] = upper;
				double eval = (uppers[d] - lowers[d]) / decisionLength[d]; 
				temp += eval;
				//mems[x] = Math.max(mems[x], eval);
			}

			mems[x] = 1-(temp/decCardinality);

			ret += temp;
			//System.out.println(x+": "+mems[x]);
		}
		//System.out.println("Denom = "+denom);
		//System.out.println("Card = "+decCardinality);
		//System.out.println("Ret = "+ret);
		//System.out.println("Blah = "+blah);		

		return 1 - (ret / denom);
	}

	public double calculate(BitSet subset) {
		generatePartition(subset);

		double lower;
		lowers = new double[m_numInstances];
		double currLower = 0;
		uppers = new double[m_numInstances];
		double ret = 0;
		double upper = 0;
		double currUpper = 0;
		double denom = 0;
		denom = (n_objects_d * decCardinality);

		// for optimization
		indexes = new int[m_numInstances];
		vals = new double[m_numInstances];

		// optimization - see if other objects in the dataset are identical/have
		// identical similarity
		setIndexes(current);

		for (int x = 0; x < m_numInstances; x++) {
			if (indexes[x] != 0) {
				vals[x] = vals[indexes[x] - 1];
				ret += vals[x];
			} else {
				for (int d = 0; d < m_numInstances; d++) {
					if (decIndexes[d] != 0) {
						lowers[d] = lowers[decIndexes[d] - 1];
					} else {
						lower = 1;

						// lower approximations of object x
						for (int y = 0; y < m_numInstances; y++) {
							double condSim = current.getCell(x, y);

							currLower = m_Implicator.calculate(
									condSim,
									fuzzySimilarity(m_classIndex, d, y));

							lower = Math.min(currLower, lower);
							if (lower == 0)
								break;
						}

						lowers[d] = lower;
					}
				}

				double temp = 0;

				for (int d = 0; d < m_numInstances; d++) {
					if (lowers[d] == 1) {
						uppers[d] = 1;
					} else {
						if (decIndexes[d] != 0) {
							uppers[d] = uppers[decIndexes[d] - 1];
						} else {
							upper = 0;

							// upper approximations of object x
							for (int y = 0; y < m_numInstances; y++) {
								double condSim = current.getCell(x, y);

								currUpper = m_TNorm.calculate(
										condSim,
										fuzzySimilarity(m_classIndex, d, y));

								upper = Math.max(currUpper, upper);
								if (upper == 1)
									break;
							}

							uppers[d] = upper;
						}

						temp += (uppers[d] - lowers[d]) / decisionLength[d];
					}
				}
				vals[x] = temp;
				ret += temp;
			}
			mems[x] = vals[x];

			// optimization - if we're not going to do better than bestGamma,
			// then quit
			// if ((ret/val)+(bestGamma*c_divisor)>1) {ret=val;break;}
		}

		return 1 - (ret / denom);
	}


	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation        result;
		TechnicalInformation        additional;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "R. Jensen, Q. Shen");
		result.setValue(Field.YEAR, "2009");
		result.setValue(Field.TITLE, "New Approaches to Fuzzy-rough Feature Selection");
		result.setValue(Field.JOURNAL, "IEEE Transactions on Fuzzy Systems");
		result.setValue(Field.SCHOOL, "Aberystwyth University");
		result.setValue(Field.ADDRESS, "");

		additional = result.add(Type.INPROCEEDINGS);
		additional.setValue(Field.AUTHOR, "N. Mac Parthalain, R. Jensen and Q. Shen");
		additional.setValue(Field.TITLE, "Finding Fuzzy-rough Reducts with Fuzzy Entropy");
		additional.setValue(Field.BOOKTITLE, "Proceedings of the 17th International Conference on Fuzzy Systems (FUZZ-IEEE'08)");
		additional.setValue(Field.YEAR, "2008");
		additional.setValue(Field.PAGES, "1282-1288");
		additional.setValue(Field.PUBLISHER, "IEEE");

		return result;
	}

	@Override
	public String globalInfo() {
		return "Subset evaluator, using the fuzzy rough boundary region.\n\n"
				+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Boundary region";
	}

	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}

