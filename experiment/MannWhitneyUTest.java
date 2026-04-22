
/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Some code used from Apache Commons Math
 */

package weka.experiment;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Statistics;
import weka.experiment.WilcoxonSignedRank.IntDoublePair;
import weka.core.Utils;

/**
 * Mann-Whitney UTest
 *
 * <p/>
 * 
 * For more information see:
 * <p/>
 * 
 * <!-- technical-plaintext-start --> Claude Nadeau, Yoshua Bengio (2001).
 * Inference for the Generalization Error. Machine Learning.. <!--
 * technical-plaintext-end -->
 * 
 * <p/>
 * 
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D &lt;index,index2-index4,...&gt;
 *  Specify list of columns that specify a unique dataset.
 *  First and last are valid indexes. (default none)
 * </pre>
 * 
 * <pre>
 * -R &lt;index&gt;
 *  Set the index of the column containing the run number
 * </pre>
 * 
 * <pre>
 * -F &lt;index&gt;
 *  Set the index of the column containing the fold number
 * </pre>
 * 
 * <pre>
 * -G &lt;index1,index2-index4,...&gt;
 *  Specify list of columns that specify a unique
 *  'result generator' (eg: classifier name and options).
 *  First and last are valid indexes. (default none)
 * </pre>
 * 
 * <pre>
 * -S &lt;significance level&gt;
 *  Set the significance level for comparisons (default 0.05)
 * </pre>
 * 
 * <pre>
 * -V
 *  Show standard deviations
 * </pre>
 * 
 * <pre>
 * -L
 *  Produce table comparisons in Latex table format
 * </pre>
 * 
 * 
 * <!-- options-end -->
 * 
 * @author Adapted from http://home.apache.org/~luc/commons-math-3.6-RC2-site/jacoco/org.apache.commons.math3.stat.inference/index.source.html by Richard Jensen
 * @version $Revision$
 */
public class MannWhitneyUTest extends PairedTTester {

	/** for serialization */
	static final long serialVersionUID = -3105268939845657324L;

	private String decimalPlaces = "%.4f";

	public void setPrecision(int numDecPlaces) {
		decimalPlaces = "%.4f";
		// uncomment this to use the number of decimal places for the mean set in the Experimenter
		//decimalPlaces = "%."+numDecPlaces+"f";
	}

	public static enum CorrectionType {
		BONFERRONI,
		NEMENYI
	}

	/** Adjust p value to reduce family-wise error rate - defaults to Bonferroni */
	private CorrectionType pCorrection = CorrectionType.BONFERRONI;

	public enum TiesStrategy {

		/** Ties assigned sequential ranks in order of occurrence */
		SEQUENTIAL,

		/** Ties get the minimum applicable rank */
		MINIMUM,

		/** Ties get the maximum applicable rank */
		MAXIMUM,

		/** Ties get the average of applicable ranks */
		AVERAGE,

		/** Ties get a random integral value from among applicable ranks */
		RANDOM
	}

	/** Ties strategy - defaults to ties averaged */
	private TiesStrategy tiesStrategy = TiesStrategy.AVERAGE;


	public StringBuffer output = new StringBuffer();
	public String prevDataset="";

	/**
	 * Computes a comparison for a specified dataset between two
	 * resultsets.
	 * 
	 * @param datasetSpecifier the dataset specifier
	 * @param resultset1Index the index of the first resultset
	 * @param resultset2Index the index of the second resultset
	 * @param comparisonColumn the column containing values to compare
	 * @return the results of the paired comparison
	 * @throws Exception if an error occurs
	 */
	@Override
	public PairedStats calculateStatistics(Instance datasetSpecifier,
			int resultset1Index, int resultset2Index, int comparisonColumn)
					throws Exception {

		if (m_Instances.attribute(comparisonColumn).type() != Attribute.NUMERIC) {
			throw new Exception("Comparison column " + (comparisonColumn + 1) + " ("
					+ m_Instances.attribute(comparisonColumn).name() + ") is not numeric");
		}
		if (!m_ResultsetsValid) {
			prepareData();
		}

		setPrecision(m_ResultMatrix.getMeanPrec());

		boolean latexOutput=false;

		if(m_ResultMatrix instanceof ResultMatrixLatex) {
			latexOutput = true;
		}
		else latexOutput=false;

		// whether to invert the performance measure
		double invert = Friedman.invert(comparisonColumn);

		Resultset resultset1 = (Resultset) m_Resultsets.get(resultset1Index);
		Resultset resultset2 = (Resultset) m_Resultsets.get(resultset2Index);

		ArrayList<Instance> dataset1 = resultset1.dataset(datasetSpecifier);
		ArrayList<Instance> dataset2 = resultset2.dataset(datasetSpecifier);

		String datasetName = templateString(datasetSpecifier);

		if (dataset1 == null) {
			throw new Exception("No results for dataset=" + datasetName
					+ " for resultset=" + resultset1.templateString());
		} else if (dataset2 == null) {
			throw new Exception("No results for dataset=" + datasetName
					+ " for resultset=" + resultset2.templateString());
		} else if (dataset1.size() != dataset2.size()) {
			throw new Exception("Results for dataset=" + datasetName
					+ " differ in size for resultset=" + resultset1.templateString()
					+ " and resultset=" + resultset2.templateString());
		}

		// calculate the test/train ratio
		double testTrainRatio = 0.0;
		int trainSizeIndex = -1;
		int testSizeIndex = -1;
		// find the columns with the train/test sizes
		for (int i = 0; i < m_Instances.numAttributes(); i++) {
			if (m_Instances.attribute(i).name().toLowerCase()
					.equals("number_of_training_instances")) {
				trainSizeIndex = i;
			} else if (m_Instances.attribute(i).name().toLowerCase()
					.equals("number_of_testing_instances")) {
				testSizeIndex = i;
			}
		}

		if (trainSizeIndex >= 0 && testSizeIndex >= 0) {
			double totalTrainSize = 0.0;
			double totalTestSize = 0.0;
			for (int k = 0; k < dataset1.size(); k++) {
				Instance current = dataset1.get(k);
				totalTrainSize += current.value(trainSizeIndex);
				totalTestSize += current.value(testSizeIndex);
			}
			testTrainRatio = totalTestSize / totalTrainSize;
		}

		// not actually used
		PairedStats pairedStats = new PairedStatsCorrected(m_SignificanceLevel,
				testTrainRatio);


		double[] x = new double[dataset1.size()]; 
		double[] y = new double[x.length];

		for (int k = 0; k < dataset1.size(); k++) {
			Instance current1 = dataset1.get(k);
			Instance current2 = dataset2.get(k);
			if (current1.isMissing(comparisonColumn)) {
				System.err.println("Instance has missing value in comparison "
						+ "column!\n" + current1);
				continue;
			}
			if (current2.isMissing(comparisonColumn)) {
				System.err.println("Instance has missing value in comparison "
						+ "column!\n" + current2);
				continue;
			}
			if (current1.value(m_RunColumn) != current2.value(m_RunColumn)) {
				System.err.println("Run numbers do not match!\n" + current1 + current2);
			}
			if (m_FoldColumn != -1) {
				if (current1.value(m_FoldColumn) != current2.value(m_FoldColumn)) {
					System.err.println("Fold numbers do not match!\n" + current1
							+ current2);
				}
			}

			double value1 = current1.value(comparisonColumn);
			double value2 = current2.value(comparisonColumn);
			pairedStats.add(value1, value2);

			value1 *= invert;
			value2 *= invert;

			// swap??
			x[k] = value2;
			y[k] = value1;
		}

		if (resultset1Index==resultset2Index) return pairedStats;

		final double[] z = concatenateSamples(x, y);
		final double[] ranks = rank(z);

		double sumRankX = 0;

		/*
		 * The ranks for x is in the first x.length entries in ranks because x
		 * is in the first x.length entries in z
		 */
		for (int i = 0; i < x.length; ++i) {
			sumRankX += ranks[i];
		}

		/*
		 * U1 = R1 - (n1 * (n1 + 1)) / 2 where R1 is sum of ranks for sample 1,
		 * e.g. x, n1 is the number of observations in sample 1.
		 */
		double U1 = sumRankX - ((long) x.length * (x.length + 1)) / 2;

		/*
		 * It can be shown that U1 + U2 = n1 * n2
		 */
		double U2 = (long) x.length * y.length - U1;


		String outp = "";
		String compare = m_Instances.attribute(comparisonColumn).name();

		if (!prevDataset.equals(datasetName)) {
			if (latexOutput) {
				if (!prevDataset.equals("")) { // this finishes off the previous table definition - no need to do this first time round
					outp+="\\hline\n" +  
							"\\end{tabular} \\footnotesize \\par}\n" + 
							"\\end{table}\n";
				}

				outp+="\n\\begin{table}[thb]\n" + 
						"\\caption{\\label{mannwhitney-"+datasetName+"} Mann-Whitney U test: evaluating for dataset "+datasetName
						+" based on " + compare.replace("_", "\\_") + " for Algorithm ("+(resultset1Index+1)+"), $\\alpha$ = "+m_SignificanceLevel+"\\\\}\n" + 
						"\\footnotesize\n" + 
						"{\\centering \\begin{tabular}{c||c|c|c|c}\n" + 
						"\\hline\n"
						+ "Comparison & $U_1$ & $U_2$ & Asymptotic p & Adjusted p \\\\"
						+ "\n\\hline\n";	
			}
			else {
				outp+="\nDataset: "+datasetName+", "+ "based on "+m_Instances.attribute(comparisonColumn).name()+"\n"
						+ "Algorithms\tU1\t\tU2\t\tAsymptotic p\tAdjusted p\n"
						+ "-----------------------------------------------------------------------------\n";
			}
		}

		prevDataset = datasetName;
		String thisComparison=datasetName+resultset1Index+resultset2Index+" ";

		int numAlgorithms = m_Resultsets.size();

		outp += performTest(U1, U2, x.length, y.length, latexOutput, resultset1Index, resultset2Index, numAlgorithms);


		pairedStats.differencesProbability=adjusted;
		pairedStats.calculateDerived();

		if (adjusted<=m_SignificanceLevel) pairedStats.differencesSignificance=1;
		else pairedStats.differencesSignificance=0;

		// Test the other way round
		outp += performTest(U2, U1, y.length, x.length, latexOutput, resultset2Index, resultset1Index, numAlgorithms);

		if (adjusted<=m_SignificanceLevel) pairedStats.differencesSignificance=-1;

		if (alreadyCompared.indexOf(thisComparison)==-1) 
			output.append(outp);

		alreadyCompared+=thisComparison; // list of comparisons

		return pairedStats;
	}

	// Hack to prevent double output in the results panel
	private String alreadyCompared="";

	public void reset() {
		prevDataset = "";
		alreadyCompared = "";
		output = new StringBuffer();
	}

	private double asymptotic;
	private double adjusted;

	/*
	 * Test across all datasets rather than all folds for a single dataset
	 * This is explicitly called in ResultsPanel
	 * 
	 * @see weka.experiment.PairedTTester#multiResultsetFull(int, int)
	 */
	public String acrossDatasets(int baseResultset, int comparisonColumn) throws Exception {
		StringBuffer result = new StringBuffer(1000);
		setPrecision(m_ResultMatrix.getMeanPrec());

		if (m_Instances.attribute(comparisonColumn).type() != Attribute.NUMERIC) {
			throw new Exception("Comparison column " + (comparisonColumn + 1) + " ("
					+ m_Instances.attribute(comparisonColumn).name() + ") is not numeric");
		}
		if (!m_ResultsetsValid) {
			prepareData();
		}

		// whether to invert the performance measure
		double invert = Friedman.invert(comparisonColumn);

		Resultset baseResultSet = (Resultset) m_Resultsets.get(baseResultset);
		int numberOfDatasets = baseResultSet.m_Datasets.size();

		Resultset resultset = (Resultset) m_Resultsets.get(0);
		ArrayList<Instance> dataset1 = resultset.dataset(m_DatasetSpecifiers.specifier(0));
		int numAlgorithms = m_Resultsets.size(); // number of algorithms
		int folds = dataset1.size();

		double[] x = new double[numberOfDatasets]; 
		double[] y = new double[x.length];
		double[][] data = new double[numberOfDatasets][numAlgorithms];
		boolean latexOutput=false;

		if(m_ResultMatrix instanceof ResultMatrixLatex) {
			latexOutput = true;
		}
		else latexOutput=false;


		// for each dataset, average the results for each algorithm
		for (int datasetIndex=0; datasetIndex<numberOfDatasets; datasetIndex++) {
			Instance datasetSpecifier = m_DatasetSpecifiers.specifier(datasetIndex);


			for (int alg = 0; alg<numAlgorithms; alg++) { // for each algorithm 
				resultset = (Resultset) m_Resultsets.get(alg);
				dataset1 = resultset.dataset(datasetSpecifier);
				data[datasetIndex][alg] = 0;

				for (int result1 = 0; result1<folds; result1++) { // for each result (fold)
					Instance current1 = dataset1.get(result1);

					if (current1.isMissing(comparisonColumn)) {
						System.err.println("Instance has missing value in comparison "
								+ "column!\n" + current1);
						continue;
					}

					data[datasetIndex][alg] += invert*current1.value(comparisonColumn);
				}

				data[datasetIndex][alg]/=folds; //get the average

				//System.out.println(alg+":  "+ data[datasetIndex][alg]);
			}
		}

		String outp="";
		String compare = m_Instances.attribute(comparisonColumn).name();

		if (latexOutput) {	
			// Hack: this finishes off the table from the individual dataset comparisons previously
			outp ="\\hline\n" +  
					"\\end{tabular} \\footnotesize \\par}\n" + 
					"\\end{table}\n\n";

			outp+="\n\\begin{table}[thb]\n" + 
					"\\caption{\\label{mann-whitney-all} Mann-Whitney U test: evaluating across all datasets, based on " 
					+ compare.replace("_", "\\_") + ", $\\alpha$ = "+m_SignificanceLevel+"\\\\}\n" + 
					"\\footnotesize\n" + 
					"{\\centering \\begin{tabular}{c||c|c|c|c}\n" + 
					"\\hline\n"
					+ "Comparison & $U_1$ & $U_2$ & Asymptotic p & Adjusted p \\\\"
					+ "\n\\hline\n";
		}
		else {
			outp="\n-------------------------------------------------------------------------------\n"
					+ "Evaluating across all datasets\n"
					+ "-------------------------------------------------------------------------------\n\nAlgorithms:\n";
			for (int i=0;i<numAlgorithms;i++) {
				resultset = (Resultset) m_Resultsets.get(i);
				outp+="("+(i+1)+") "+resultset.templateString()+"\n";
			}

			outp += "\nFull comparison of \n("+(baseResultset+1)+") "+baseResultSet.templateString()+"\nwith all other algorithms, ";
			outp += "based on: "+compare+ ", alpha = "+m_SignificanceLevel+"\n";

			outp+= "\nComparison\tU1\t\tU2\t\tAsymptotic p\tAdjusted p\n";
			outp+="-----------------------------------------------------------------------------\n";
		}

		result.append(outp);

		// for each algorithm, compare with baseResultset
		for (int alg=0; alg<numAlgorithms; alg++) {
			if (alg==baseResultset) continue;

			for (int d=0;d<numberOfDatasets;d++) {
				x[d]=data[d][alg];
				y[d]=data[d][baseResultset];
			}

			final double[] z = concatenateSamples(x, y);
			final double[] ranks = rank(z);

			double sumRankX = 0;

			/*
			 * The ranks for x is in the first x.length entries in ranks because x
			 * is in the first x.length entries in z
			 */
			for (int i = 0; i < x.length; ++i) {
				sumRankX += ranks[i];
			}

			String summary="";


			/*
			 * U1 = R1 - (n1 * (n1 + 1)) / 2 where R1 is sum of ranks for sample 1,
			 * e.g. x, n1 is the number of observations in sample 1.
			 */
			double U1 = sumRankX - ((long) x.length * (x.length + 1)) / 2;

			/*
			 * It can be shown that U1 + U2 = n1 * n2
			 */
			double U2 = (long) x.length * y.length - U1;


			summary += performTest(U1, U2, x.length, y.length, latexOutput, baseResultset, alg, numAlgorithms);

			// test the other way
			summary += performTest(U2, U1, y.length, x.length, latexOutput, alg, baseResultset, numAlgorithms);

			result.append(summary);

		}

		if (latexOutput) {
			result.append("\\hline\n" +  
					"\\end{tabular} \\footnotesize \\par}\n" + 
					"\\end{table}");
		}
		else result.append("-----------------------------------------------------------------------------\n\n");

		output = result;
		return "";//result.toString();
	}

	/**
	 * Performs the test. Also sets the value of private variables: asymptotic and adjusted.
	 * @param U1 U1
	 * @param U2 U2
	 * @param xlength size of the x array
	 * @param ylength size of the y array
	 * @param latexOutput whether LaTeX output is required
	 * @param baseResultset index of the first algorithm
	 * @param alg index of the second algorithm
	 * @param numAlgorithms number of algorithms
	 * @return report of the test
	 */
	private String performTest(double U1, double U2, int xlength, int ylength, boolean latexOutput, int baseResultset, int alg, int numAlgorithms) {
		String summary="";

		if (latexOutput) {
			summary = "("+(baseResultset+1)+ ") vs ("+(alg+1)+") "+ " & "+U1+" & "+U2+" & ";
		}
		else {
			summary = "("+(baseResultset+1)+ ") vs ("+(alg+1)+") "+ "\t"+U1+"\t\t"+U2+"\t\t";

		}

		asymptotic =  calculateAsymptoticPValue(U2, xlength, ylength);

		// Correction to reduce family-wise error rate
		adjusted = correctPValue(numAlgorithms,asymptotic);

		// is this result significant?
		boolean significant = adjusted<m_SignificanceLevel;

		if (latexOutput) {
			summary += String.format(decimalPlaces, asymptotic)+" & "+(significant? "{\\bf ":"")+String.format(decimalPlaces, adjusted)+(significant? "}":"")+" \\\\ \n";
		}
		else summary += String.format(decimalPlaces, asymptotic)+"\t\t"+String.format(decimalPlaces, adjusted)+(significant? " *":" ")+"\n";

		return summary;
	}

	/** Concatenate the samples into one array.
	 * @param x first sample
	 * @param y second sample
	 * @return concatenated array
	 */
	private double[] concatenateSamples(final double[] x, final double[] y) {
		final double[] z = new double[x.length + y.length];

		System.arraycopy(x, 0, z, 0, x.length);
		System.arraycopy(y, 0, z, x.length, y.length);

		return z;
	}


	/**
	 * Correct the p value to reduce family-wise error rate.
	 * @param algorithms the number of algorithms
	 * @param unadjustedP the unadjusted p value
	 * @return the corrected p value
	 */
	public double correctPValue(int algorithms, double unadjustedP) {
		double adjusted = unadjustedP;

		switch (pCorrection) {
		case BONFERRONI:
			adjusted = Math.min(1, unadjustedP*(algorithms-1));
			break;
		case NEMENYI:
			adjusted = Math.min(1, unadjustedP*combination(2,algorithms));
			break;
		}
		return adjusted;
	}

	/**
	 * @param Umin smallest Mann-Whitney U value
	 * @param n1 number of subjects in first sample
	 * @param n2 number of subjects in second sample
	 * @return two-sided asymptotic p-value
	 * @throws ConvergenceException if the p-value can not be computed
	 * due to a convergence error
	 * @throws MaxCountExceededException if the maximum number of
	 * iterations is exceeded
	 */
	private double calculateAsymptoticPValue(final double Umin,
			final int n1,
			final int n2) {

		/* long multiplication to avoid overflow (double not used due to efficiency
		 * and to avoid precision loss)
		 */
		final long n1n2prod = (long) n1 * n2;

		// http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
		final double EU = n1n2prod / 2.0;
		final double VarU = n1n2prod * (n1 + n2 + 1) / 12.0;

		final double z = (Umin - EU) / Math.sqrt(VarU);

		return Math.min(1, 2 * Statistics.normalProbability(z));
	}


	/**
	 * Rank <code>data</code> using the natural ordering on Doubles, with ties
	 * resolved using <code>tiesStrategy.</code>
	 *
	 * @param data array to be ranked
	 * @return array of ranks
	 */
	public double[] rank(double[] data) {

		// Array recording initial positions of data to be ranked
		IntDoublePair[] ranks = new IntDoublePair[data.length];

		for (int i = 0; i < data.length; i++) {
			ranks[i] = new IntDoublePair(data[i], i);
		}


		// Sort the IntDoublePairs
		Arrays.sort(ranks);

		// Walk the sorted array, filling output array using sorted positions,
		// resolving ties as we go
		double[] out = new double[ranks.length];
		int pos = 1;  // position in sorted array
		out[ranks[0].getPosition()] = pos;

		List<Integer> tiesTrace = new ArrayList<Integer>();
		tiesTrace.add(ranks[0].getPosition());
		for (int i = 1; i < ranks.length; i++) {
			if (Double.compare(ranks[i].getValue(), ranks[i - 1].getValue()) > 0) {
				// tie sequence has ended (or had length 1)
				pos = i + 1;
				if (tiesTrace.size() > 1) {  // if seq is nontrivial, resolve
					resolveTie(out, tiesTrace);
				}
				tiesTrace = new ArrayList<Integer>();
				tiesTrace.add(ranks[i].getPosition());
			} else {
				// tie sequence continues
				tiesTrace.add(ranks[i].getPosition());
			}
			out[ranks[i].getPosition()] = pos;
		}
		if (tiesTrace.size() > 1) {  // handle tie sequence at end
			resolveTie(out, tiesTrace);
		}

		return out;
	}

	Random random = new Random();

	/**
	 * Resolve a sequence of ties, using the configured {@link TiesStrategy}.
	 * The input <code>ranks</code> array is expected to take the same value
	 * for all indices in <code>tiesTrace</code>.  The common value is recoded
	 * according to the tiesStrategy. For example, if ranks = <5,8,2,6,2,7,1,2>,
	 * tiesTrace = <2,4,7> and tiesStrategy is MINIMUM, ranks will be unchanged.
	 * The same array and trace with tiesStrategy AVERAGE will come out
	 * <5,8,3,6,3,7,1,3>.
	 *
	 * @param ranks array of ranks
	 * @param tiesTrace list of indices where <code>ranks</code> is constant
	 * -- that is, for any i and j in TiesTrace, <code> ranks[i] == ranks[j]
	 * </code>
	 */
	private void resolveTie(double[] ranks, List<Integer> tiesTrace) {

		// constant value of ranks over tiesTrace
		final double c = ranks[tiesTrace.get(0)];

		// length of sequence of tied ranks
		final int length = tiesTrace.size();

		switch (tiesStrategy) {
		case  AVERAGE:  // Replace ranks with average
			fill(ranks, tiesTrace, (2 * c + length - 1) / 2d);
			break;
		case MAXIMUM:   // Replace ranks with maximum values
			fill(ranks, tiesTrace, c + length - 1);
			break;
		case MINIMUM:   // Replace ties with minimum
			fill(ranks, tiesTrace, c);
			break;
		case RANDOM:    // Fill with random integral values in [c, c + length - 1]
			Iterator<Integer> iterator = tiesTrace.iterator();
			long f = Math.round(c);
			while (iterator.hasNext()) {
				// No advertised exception because args are guaranteed valid
				ranks[iterator.next()] = 
						random.nextLong(f, f + length - 1);
			}
			break;
		case SEQUENTIAL:  // Fill sequentially from c to c + length - 1
			// walk and fill
			iterator = tiesTrace.iterator();
			f = Math.round(c);
			int i = 0;
			while (iterator.hasNext()) {
				ranks[iterator.next()] = f + i++;
			}
			break;

		}
	}

	/**
	 * Sets<code>data[i] = value</code> for each i in <code>tiesTrace.</code>
	 *
	 * @param data array to modify
	 * @param tiesTrace list of index values to set
	 * @param value value to set
	 */
	private void fill(double[] data, List<Integer> tiesTrace, double value) {
		Iterator<Integer> iterator = tiesTrace.iterator();
		while (iterator.hasNext()) {
			data[iterator.next()] = value;
		}
	}


	/**
	 * Test the class from the command line.
	 * 
	 * @param args contains options for the instance ttests
	 */
	public static void main(String args[]) {

		try {
			MannWhitneyUTest tt = new MannWhitneyUTest();
			String datasetName = Utils.getOption('t', args);
			String compareColStr = Utils.getOption('c', args);
			String baseColStr = Utils.getOption('b', args);
			boolean summaryOnly = Utils.getFlag('s', args);
			boolean rankingOnly = Utils.getFlag('r', args);
			try {
				if ((datasetName.length() == 0) || (compareColStr.length() == 0)) {
					throw new Exception("-t and -c options are required");
				}
				tt.setOptions(args);
				Utils.checkForRemainingOptions(args);
			} catch (Exception ex) {
				String result = "";
				Enumeration<Option> enu = tt.listOptions();
				while (enu.hasMoreElements()) {
					Option option = enu.nextElement();
					result += option.synopsis() + '\n' + option.description() + '\n';
				}
				throw new Exception("Usage:\n\n" + "-t <file>\n"
						+ "\tSet the dataset containing data to evaluate\n" + "-b <index>\n"
						+ "\tSet the resultset to base comparisons against (optional)\n"
						+ "-c <index>\n" + "\tSet the column to perform a comparison on\n"
						+ "-s\n" + "\tSummarize wins over all resultset pairs\n\n" + "-r\n"
						+ "\tGenerate a resultset ranking\n\n" + result);
			}
			Instances data = new Instances(new BufferedReader(new FileReader(
					datasetName)));
			tt.setInstances(data);
			// tt.prepareData();
			int compareCol = Integer.parseInt(compareColStr) - 1;
			System.out.println(tt.header(compareCol));
			if (rankingOnly) {
				System.out.println(tt.multiResultsetRanking(compareCol));
			} else if (summaryOnly) {
				System.out.println(tt.multiResultsetSummary(compareCol));
			} else {
				System.out.println(tt.resultsetKey());
				if (baseColStr.length() == 0) {
					for (int i = 0; i < tt.getNumResultsets(); i++) {
						System.out.println(tt.multiResultsetFull(i, compareCol));
					}
				} else {
					int baseCol = Integer.parseInt(baseColStr) - 1;
					System.out.println(tt.multiResultsetFull(baseCol, compareCol));
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

	/**
	 * Computes the (N/M) combinatory number
	 *
	 * @param n N value
	 * @param m M value
	 *
	 * @return The (N/M) combinatory number
	 */
	public static double combination(int m, int n) {
		double result = 1;
		int i;

		if (n >= m) {
			for (i=1; i<=m; i++)
				result *= (double)(n-m+i)/(double)i;
		} else {
			result = 0;
		}
		return result;
	}

	/**
	 * returns the name of the tester
	 * 
	 * @return the display name
	 */
	@Override
	public String getDisplayName() {
		return "Mann-Whitney U Test";
	}

	/**
	 * returns a string that is displayed as tooltip on the "perform test" button
	 * in the experimenter
	 * 
	 * @return the string for the tool tip
	 */
	@Override
	public String getToolTipText() {
		return "Performs test using Mann-Whitney U Test";
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}

}
