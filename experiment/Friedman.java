package weka.experiment;

import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.OptionHandler;
import weka.core.RevisionHandler;
import weka.core.Statistics;


// TODO: Auto-generated Javadoc
/**
 * Performs a Friedman test on a data table
 *
 * The Friedman Test is a non-parametric statistical test used to determine if
 * there is statistical significance in results between three or more samples.
 * 
 * Calculating a Friedman Test involves ensuring that the data read in from a
 * file is transformed into ranks where the smallest value equals the higher
 * rank
 * 
 * Code by Kieran Stone (and Richard Jensen)
 */

public class Friedman extends PairedTTester implements Tester, OptionHandler, RevisionHandler {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 721564540689253548L;

	/** The data. */
	double[][] data;

	static String[] algorithmNames;

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.experiment.PairedTTester#getDisplayName()
	 */
	public String getDisplayName() {
		return "Friedman Test";
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.experiment.PairedTTester#getToolTipText()
	 */
	public String getToolTipText() {
		// TODO Auto-generated method stub
		return "Two Way Analysis of Variance by Ranks";
	}

	private static String decimalPlaces = "%.6f"; // 6 decimal places

	public void setPrecision(int numDecPlaces) {
		decimalPlaces = "%.6f";
		// uncomment this to use the number of decimal places for the mean set in the Experimenter
		//decimalPlaces = "%."+numDecPlaces+"f";
	}


	/**
	 * Inverts the performance measure. Needed for measures where higher values mean better performance (e.g. accuracy)
	 * @param compColumn the performance comparison column
	 * @return -1 if compColumn is a measure that needs to be inverted
	 */
	public static double invert(int compColumn) {
		// cases where the measure is better the higher it is
		// compColumn==5 for use in the converter program that converts from CSV ro ARFF
		if ((compColumn==5)||(compColumn==9)||(compColumn==12)||(compColumn==15)||(compColumn==29)||(compColumn==30)||(compColumn==33)||(compColumn==34)||((compColumn>=37)&&(compColumn<=43))
				||(compColumn==45)||((compColumn>=47)&&(compColumn==54))||(compColumn==67)) return -1;
		else return 1;
	}
	
	private String holmLabel="";

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.experiment.PairedTTester#multiResultsetFull(int, int)
	 */
	public String multiResultsetFull(int baseResultset, int comparisonColumn) throws Exception {
		StringBuffer result = new StringBuffer(1000);
		setPrecision(m_ResultMatrix.getMeanPrec());

		if (m_Instances.attribute(comparisonColumn).type() != Attribute.NUMERIC) {
			throw new Exception("Comparison column " + (comparisonColumn + 1) + " ("
					+ m_Instances.attribute(comparisonColumn).name() + ") is not numeric");
		}
		if (!m_ResultsetsValid) {
			prepareData();
		}

		// prints out all performance measures and their index
		//for (int i=0; i<m_Instances.numAttributes();i++) System.err.println(i+": "+m_Instances.attribute(i).name());

		// decide whether to invert the performance measure
		double invert = invert(comparisonColumn);

		Resultset resultset1 = (Resultset) m_Resultsets.get(baseResultset);
		int numberOfDatasets = resultset1.m_Datasets.size();

		Resultset resultset = (Resultset) m_Resultsets.get(0);
		ArrayList<Instance> dataset1 = resultset.dataset(m_DatasetSpecifiers.specifier(0));

		int columns = m_Resultsets.size(); // number of algorithms
		if (columns<3) return "\nError: must be 3 or more algorithms\n";

		int rows = dataset1.size();

		if(m_ResultMatrix instanceof ResultMatrixLatex) {
			latexOutput = true;
		}
		else latexOutput=false;


		// will store the average performance measure for each dataset for each algorithm
		double[][] acrossDatasets = new double[numberOfDatasets][columns];

		result.append("\nAlgorithms:\n");
		algorithmNames = new String[columns];

		for (int i=0;i<columns;i++) {
			resultset = (Resultset) m_Resultsets.get(i);
			String desc = resultset.templateString();
			result.append("("+(i+1)+") "+desc+"\n");
			algorithmNames[i] = desc.substring(desc.indexOf(".")+1,desc.indexOf(" "));
			System.out.println(algorithmNames[i]);
		}

		// for each dataset, convert the data into a 2D double array, which is then used to perform the test
		for (int datasetIndex=0; datasetIndex<numberOfDatasets; datasetIndex++) {
			Instance datasetSpecifier = m_DatasetSpecifiers.specifier(datasetIndex);
			String compare = m_Instances.attribute(comparisonColumn).name();
			String datasetName = templateString(datasetSpecifier);
			holmLabel=""; // used to create a unique label for the output for Holm
			
			if (latexOutput) {
				holmLabel="-"+datasetName;
				String outp="\n\\begin{table}[thb]\n" + 
						"\\caption{\\label{friedman-"+datasetName+"} Friedman test: mean column ranks, based on " 
						+ compare.replace("_", "\\_") +"\\\\}\n" + 
						"\\footnotesize\n" + 
						"{\\centering \\begin{tabular}{c|c}\n" + 
						"\\hline\n"
						+ "Algorithm"+"& Ranking \\\\"
						+ "\n\\hline\n";	
				result.append(outp);
			}
			else {
				result.append("\n--------------------------------------------------\n");
				result.append("Friedman test for: "+datasetName);
				result.append("\nbased on "+compare+"\n");
			}

			// this will store the raw data
			data = new double[rows][columns];

			// convert into 2D data
			for (int index = 0; index<columns; index++) { // for each algorithm 
				resultset = (Resultset) m_Resultsets.get(index);
				dataset1 = resultset.dataset(datasetSpecifier);

				for (int result1=0; result1<rows; result1++) { // for each result (fold)
					Instance current1 = dataset1.get(result1);

					if (current1.isMissing(comparisonColumn)) {
						System.err.println("Instance has missing value in comparison "
								+ "column!\n" + current1);
						continue;
					}

					data[result1][index] = invert*current1.value(comparisonColumn);
					acrossDatasets[datasetIndex][index] += data[result1][index];
				}

				acrossDatasets[datasetIndex][index]/=rows;

			}

			// compute ranks (within each row)
			double[][] rank = new double[rows][columns];
			for (int i = 0; i < rank.length; i++) {
				// get ranks at row i
				double[] r = getRanks(data[i]);
				// copy into rank array at row i
				for (int j = 0; j < rank[i].length; j++)
					rank[i][j] = r[j];
			}


			result.append(calculateFriedmanStatistics(rank));
		}


		result.append("\n-------------------------------------------------------------------------------\n"
				+ "Friedman: evaluating across all datasets"
				+ "\n-------------------------------------------------------------------------------\n"
				+ "");

		if (latexOutput) {
			String compare = m_Instances.attribute(comparisonColumn).name();

			String outp="\n\\begin{table}[thb]\n" + 
					"\\caption{\\label{friedman-ranks} Friedman: mean column ranks, based on " 
					+ compare.replace("_", "\\_") +"\\\\}\n" + 
					"\\footnotesize\n" + 
					"{\\centering \\begin{tabular}{c|c}\n" + 
					"\\hline\n"
					+ "Algorithm"+"& Ranking \\\\"
					+ "\n\\hline\n";	
			result.append(outp);
		}


		rows = numberOfDatasets; // for testing across all datasets

		// compute ranks (within each row)
		double[][] rank = new double[rows][columns];
		for (int i = 0; i < rank.length; i++) {
			// get ranks at row i
			double[] r = getRanks(acrossDatasets[i]);
			// copy into rank array at row i
			for (int j = 0; j < rank[i].length; j++)
				rank[i][j] = r[j];
		}

		holmLabel = "";
		result.append(calculateFriedmanStatistics(rank));

		return result.toString();
	}

	boolean latexOutput=false;

	/**
	 * Calculate Friedman statistics.
	 * 
	 * Calculates Friedman's F statistic and makes use of weka.core statistics
	 */
	public String calculateFriedmanStatistics(double[][] rank) {
		StringBuffer outputResultStatistics= new StringBuffer();
		double[] columnSum = new double[rank[0].length];
		for (int i = 0; i < rank[0].length; ++i) // 'i' iterates through columns
		{
			for (int j = 0; j < rank.length; ++j) // 'j' iterates through rows
			{
				columnSum[i] += rank[j][i];
			}
		}

		int r = rank.length; // number of rows
		int c = rank[0].length; // number of algorithms
		double ss = 0.0;
		for (int i = 0; i < rank[0].length; ++i)
			ss += columnSum[i] * columnSum[i];
		double correction = correctionForTies(rank);

		if (!latexOutput) {
			outputResultStatistics.append("b = " + r + " (number of rows)"+"\n");
			outputResultStatistics.append("k = " + c + " (number of algorithms/columns)"+"\n\n");
		}

		double h = (12 * ss) / (r * c * (c + 1)) - 3.0 * r * (c + 1);
		double p = Statistics.chiSquaredProbability(h, c - 1);

		if (!latexOutput) outputResultStatistics.append(("Without correction for ties: H = "+String.format(decimalPlaces, h)+", p = "+String.format(decimalPlaces, p))+(c-1)+"\n");

		h = h / correction; // re-compute using correction for ties
		p = Statistics.chiSquaredProbability(h, c - 1);
		
		if (!latexOutput) {
			outputResultStatistics.append(("Corrected for ties: H' = "+String.format(decimalPlaces, h)+", p' = "+String.format(decimalPlaces, p))+(c-1)+"\n\n");
			outputResultStatistics.append("\nMean column ranks:\n");
		}
		
		int rows = rank.length;
		int columns = rank[0].length;
		double[] meanColRank = new double[columns];
		for (int i = 0; i < columns; ++i) {
			meanColRank[i]=0;
			for (int j = 0; j < rows; ++j) {
				meanColRank[i] += rank[j][i];
			}
			meanColRank[i] /= rows;

			if (latexOutput) outputResultStatistics.append("("+(i+1)+") & "+ meanColRank[i]+"\\\\ \n");
			else outputResultStatistics.append("Algorithm ("+(i+1)+")"+": "+ meanColRank[i]+"\n");
		}

		if (latexOutput) {
			outputResultStatistics.append("\\hline\n" +  
					"\\end{tabular} \\footnotesize \\par}\n" + 
					"\\end{table}\n\n");
		}
		else  outputResultStatistics.append("\n----------------------------------\n");

		if (latexOutput) {		
			String outp="\\begin{table}[thb]\n" + 
					"\\caption{\\label{holm"+holmLabel+"} Holm's post hoc test, $\\alpha$  = "+m_SignificanceLevel+"\\\\}\n" + 
					"\\footnotesize\n" + 
					"{\\centering \\begin{tabular}{c||c|c|c}\n" + 
					"\\hline\n"
					+ "Comparison vs. & Asymptotic p & Adj. $\\alpha$ & Significance \\\\"
					+ "\n\\hline\n";
			outputResultStatistics.append(outp);
		}


		outputResultStatistics.append(holm(meanColRank, rows, m_SignificanceLevel));

		if (latexOutput) {
			outputResultStatistics.append("\\hline\n" +  
					"\\end{tabular} \\footnotesize \\par}\n" + 
					"\\end{table}\n\n");
		}

		return outputResultStatistics.toString();
	}

	/**
	 * 
	 * If the ranking of values results in any ties, a ties correction is
	 * required. This is calculated below.
	 *
	 * @param r
	 *            the r
	 * @return the double
	 */
	protected static double correctionForTies(double[][] r) {
		double c = 0.0; // correction for ties
		double factor = 0.0;

		for (int i = 0; i < r.length; ++i) {
			double[] row = new double[r[0].length];
			for (int j = 0; j < r[i].length; ++j)
				row[j] = r[i][j];

			Arrays.sort(row);

			for (int j = 0; j < row.length; ++j) {
				double rank = row[j];
				int k = j + 1;
				int countRanks = 1;
				while (k < row.length) {
					if (rank == row[k]) {
						++countRanks;
						++j;
						++k;
					} else
						break;
				}
				if (countRanks > 1)
					factor += countRanks * countRanks * countRanks - countRanks;
			}
		}
		c = 1.0 - (factor) / (r.length * (r[0].length * r[0].length * r[0].length - r[0].length));
		return c;
	}

	/**
	 * Gets the ranks.
	 *
	 * @param data2
	 *            the data 2
	 * @return an Array of ranks
	 */

	protected static double[] getRanks(double[] data2) {
		// put raw ranks in 'r' array
		double[] r = new double[data2.length];

		// create a copy of 'd' array
		double[] a = new double[data2.length];
		for (int i = 0; i < a.length; ++i)
			a[i] = data2[i];

		Arrays.sort(a); // sort, but leave values intact

		for (int i = 0; i < a.length; ++i) {
			double rawScore = a[i];
			int rawRank = i + 1;
			int count = 1;
			int j = i + 1;
			while (j < a.length) // look for ties
			{
				if (rawScore == a[j]) // found a tie
				{
					rawRank += j + 1;
					++count;
					++j;
				} else
					// no tie, skip to next position in array
					break;
			}
			double rank = (double) rawRank / count;
			r[i] = rank;
			while (--count > 0) // fill slots with ties
			{
				++i;
				r[i] = rank;
			}
		}

		// use a (raw scores, sorted), r (ranks, sorted), and d (raw scores,
		// unsorted) to build rr
		double[] rr = new double[data2.length];

		for (int i = 0; i < data2.length; ++i) {
			for (int j = 0; j < data2.length; ++j) {
				if (data2[i] == a[j]) {
					rr[i] = r[j];
					break;
				}
			}
		}
		return rr;
	}


	/**
	 * For testing
	 *
	 */
	public static void main(String[] args) {

		Friedman friedman = new Friedman();

		//double[] algperf = {0.9, 0.22, 0.34, 0.64, 0.64}; // rank: [5.0, 1.0, 2.0, 3.5, 3.5]
		//System.err.println(Arrays.toString(Friedman.getRanks(algperf)));

		//double[][] data = {{0.9,0.22,0.34,0.64,0.64},{0.5, 0.3, 0.7, 0.4, 0.5}};

		//https://www.statology.org/friedman-test/
		/*double[][] data = {
				{4,5,2}, 
				{6,6,4}, 
				{3,8,4}, 
				{4,7,3}, 
				{3,7,2}, 
				{2,8,2}, 
				{2,4,1}, 
				{7,6,4}, 
				{6,4,3}, 
				{5,5,2}
		};*/

		//https://sci2s.ugr.es/sites/default/files/files/TutorialsAndPlenaryTalks/INIT-AERFAI-Course-Statistical_Analysis_of_Experiments.pdf
		double[][] data = {
				{0.763, 0.768, 0.771, 0.798},
				{0.599, 0.591, 0.590, 0.569},
				{0.954, 0.971, 0.968, 0.967},
				{0.628, 0.661, 0.654, 0.657},
				{0.882, 0.888, 0.886, 0.898},
				{0.936, 0.931, 0.916, 0.931},
				{0.661, 0.668, 0.609, 0.685},
				{0.583, 0.583, 0.563, 0.625},
				{0.775, 0.838, 0.866, 0.875},
				{1.000, 1.000, 1.000, 1.000},
				{0.940, 0.962, 0.965, 0.962},
				{0.619, 0.666, 0.614, 0.669},
				{0.972, 0.981, 0.975, 0.975},
				{0.957, 0.978, 0.946, 0.970}};

		// invert if needed (when higher values = good)
		for (int i=0;i<data.length;i++) for (int j=0;j<data[0].length;j++) {
			double val = data[i][j];
			data[i][j] = 1-val;
		}

		// compute ranks (within each row)
		double[][] rank = new double[data.length][data[0].length];
		for (int i = 0; i < rank.length; i++) {
			// get ranks at row i
			double[] r = getRanks(data[i]);
			// copy into rank array at row i
			for (int j = 0; j < rank[i].length; j++)
				rank[i][j] = r[j];
		}


		System.out.println(friedman.calculateFriedmanStatistics(rank));

		//for (int i=0;i<rank.length;i++) System.out.println(Arrays.toString(rank[i])+"\n");
		//System.err.println(2*(1-Statistics.normalProbability(5.471)));

		// LAIR paper
		double[] rankings = {2.1471,
				5.5441,
				5.6029,
				6.0735,
				6.9265,
				7.0441,
				7.1324,
				7.7941,
				7.8235,
				8.0147,
				8.2059,
				9.3088,
				9.3824};

		algorithmNames = new String[rankings.length];
		algorithmNames[0] = "FRPS";
		algorithmNames[1] = "LAIR";
		algorithmNames[2] = "MSS";
		algorithmNames[3] = "FRIS";
		algorithmNames[4] = "CCIS";
		algorithmNames[5] = "C-Pruner";
		algorithmNames[6] = "DROP3";
		algorithmNames[7] = "CNN";
		algorithmNames[8] = "ICF";
		algorithmNames[9] = "IB3";
		algorithmNames[10] = "FCNN";
		algorithmNames[11] = "MCNN";
		algorithmNames[12] = "Recon.";

		System.out.println(friedman.holm(rankings, 34, 0.05));
	}

	/**
	 * Holm's post hoc test. 
	 * Compares every pair of algorithms
	 */
	public String holm(double[] columnRanks, int numDatasets, double sig) {
		String report="";
		int numAlgorithms = columnRanks.length;

		IntDoublePair[] list = new IntDoublePair[numAlgorithms*(numAlgorithms-1)/2];

		double standardError = Math.sqrt((numAlgorithms*(numAlgorithms+1))/(6.0*numDatasets));

		if (!latexOutput) {
			report+="Holm's post hoc test, alpha = "+m_SignificanceLevel+"\n";
			report+="Standard error = "+standardError+"\n\n";
		}

		int index=0;
		boolean useNames=false; // set this to true to use the classifier names in the output 

		if (algorithmNames==null) useNames=false;

		// pairwise comparison of all algorithms
		for (int controlAlgorithm=0;controlAlgorithm<numAlgorithms;controlAlgorithm++) {
			for (int i=controlAlgorithm+1;i<numAlgorithms;i++) {				
				double value = Math.abs(columnRanks[controlAlgorithm]-columnRanks[i])/standardError;
				list[index] = new IntDoublePair(Math.min(2*(1-Statistics.normalProbability(value)),1),value,controlAlgorithm,i);
				index++;
			}
		}

		Arrays.sort(list);

		String significant = "significant";
		String notSignificant = "not significant";

		for (int i=0;i<list.length;i++) {
			//double adjustedAlpha = sig/(double)(numAlgorithms-(i+1));
			double adjustedAlpha = sig/(list.length-(i));

			if (latexOutput) {
				if (!useNames) report+="("+(list[i].alg1+1)+") vs. ("+(list[i].alg2+1)+")";
				else report+=algorithmNames[list[i].alg1]+" vs "+algorithmNames[list[i].alg2];

				report +=" & "+String.format(decimalPlaces, list[i].getValue())+" & "+String.format(decimalPlaces, adjustedAlpha);

				if (list[i].getValue()<=adjustedAlpha) report+= "& "+significant;
				else report += "& "+notSignificant;
				report+="\\\\ \n";
			}
			else {
				if (!useNames) report+="Algorithm ("+(list[i].alg1+1)+") vs. ("+(list[i].alg2+1)+")";
				else report+= algorithmNames[list[i].alg1]+" vs "+algorithmNames[list[i].alg2]+"";

				report +="\tp: "+String.format(decimalPlaces, list[i].getValue())+"\talpha: "+String.format(decimalPlaces, adjustedAlpha);
				//report+=" Z: "+list[i].Z;

				if (list[i].getValue()<=adjustedAlpha) report+= "\t "+significant+"\n";
				else report += "\t "+notSignificant+"\n";
			}
		}

		return report;
	}

	/**
	 * Represents the position of a double value in an ordering.
	 * Comparable interface is implemented so Arrays.sort can be used
	 * to sort an array of IntDoublePairs by value.  Note that the
	 * implicitly defined natural ordering is NOT consistent with equals.
	 */
	private static class IntDoublePair implements Comparable<IntDoublePair>  {

		/** Value of the pair */
		public final double value;

		public final double Z;

		public final int alg1;
		public final int alg2;

		/**
		 * Construct an IntDoublePair with the given value and position.
		 * @param value the value of the pair
		 * @param position the original position
		 */
		IntDoublePair(double value, double z, int a1, int a2) {
			this.value = value;
			this.Z = z;
			this.alg1 = a1;
			this.alg2 = a2;
		}

		/**
		 * Compare this IntDoublePair to another pair.
		 * Only the <strong>values</strong> are compared.
		 *
		 * @param other the other pair to compare this to
		 * @return result of <code>Double.compare(value, other.value)</code>
		 */
		public int compareTo(IntDoublePair other) {
			return Double.compare(value, other.value);
		}

		// N.B. equals() and hashCode() are not implemented; see MATH-610 for discussion.

		/**
		 * Returns the value of the pair.
		 * @return value
		 */
		public double getValue() {
			return value;
		}

	}
}