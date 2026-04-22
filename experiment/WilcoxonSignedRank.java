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


import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;

import weka.core.Attribute;
import weka.core.Instance;

import weka.core.RevisionUtils;
import weka.core.Statistics;

/**
 * Wilcoxon signed rank test
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
 *  Specify list of columns that specify a unique
 *  dataset.
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
 * <pre>
 * -csv
 *  Produce table comparisons in CSV table format
 * </pre>
 * 
 * <pre>
 * -html
 *  Produce table comparisons in HTML table format
 * </pre>
 * 
 * <pre>
 * -significance
 *  Produce table comparisons with only the significance values
 * </pre>
 * 
 * <pre>
 * -gnuplot
 *  Produce table comparisons output suitable for GNUPlot
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Kieran Stone and Richard Jensen
 * @version $Revision$
 */
public class WilcoxonSignedRank extends PairedTTester  {

	/** for serialization */
	static final long serialVersionUID = -3105268939845656323L;

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
	private static String decimalPlaces = "%.4f"; // 4 decimal places

	private int numDecPlaces=4;

	public void setPrecision(int ndp) {
		numDecPlaces = ndp;
		decimalPlaces = "%.4f";

		// uncomment this to use the number of decimal places for the mean set in the Experimenter
		//decimalPlaces = "%."+numDecPlaces+"f";
	}

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

		// whether to invert the performance measure
		double invert = Friedman.invert(comparisonColumn);

		Resultset resultset1 = (Resultset) m_Resultsets.get(resultset1Index);
		Resultset resultset2 = (Resultset) m_Resultsets.get(resultset2Index);

		ArrayList<Instance> dataset1 = resultset1.dataset(datasetSpecifier);
		ArrayList<Instance> dataset2 = resultset2.dataset(datasetSpecifier);
		boolean latexOutput=false;

		if(m_ResultMatrix instanceof ResultMatrixLatex) {
			latexOutput = true;
		}
		else latexOutput=false;

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


		double[] z = new double[dataset1.size()]; // differences
		double[] zAbs = new double[z.length]; // absolute differences
		int numZeros = 0;

		for (int k = 0; k < dataset1.size(); k++) {
			Instance current1 = dataset1.get(k);
			Instance current2 = dataset2.get(k);
			if (current1.isMissing(comparisonColumn)) {
				System.err.println("Instance has missing value in comparison column!\n" + current1);
				continue;
			}
			if (current2.isMissing(comparisonColumn)) {
				System.err.println("Instance has missing value in comparison column!\n" + current2);
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

			double diff = value2-value1;
			if (diff==0) numZeros++;
			z[k] = diff;
			zAbs[k] = Math.abs(z[k]);
		}

		if (resultset1Index==resultset2Index) return pairedStats;


		// identical performance
		if (numZeros==z.length) {
			pairedStats.differencesSignificance=0;
			return pairedStats;
		}

		// for testing
		//z = new double[] {-6,-2,-6,-4,-3,-3,3,3,-10,-10,-1,-10,-4,-10,-5,-10,-7,-7,-8};
		//zAbs = new double[] {6,2,6,4,3,3,3,3,10,10,1,10,4,10,5,10,7,7,8};

		//z = new double[] {10,20,-10,25,60,10,15,-5};
		//zAbs = new double[] {10,20,10,25,60,10,15,5};

		double[] ranks = rank(zAbs); 

		double Wplus = 0;
		double Wminus = 0;


		boolean flip = true;
		boolean oddNumberOfZeros = (numZeros%2!=0);
		boolean ignoreFirst = oddNumberOfZeros;

		for (int i = 0; i < z.length; ++i) {
			if (z[i] > 0) Wplus += ranks[i];
			else if (z[i] < 0) Wminus += ranks[i];
			else { // zero difference
				if (ignoreFirst) {	// if odd number of zeros, then ignore one result		
					ignoreFirst = false;
				}
				else {
					if (flip) Wplus += ranks[i];
					else Wminus += ranks[i];

					flip = !flip;
				}
			}
		}

		int N = z.length;
		if (oddNumberOfZeros) N -= 1;

		/**
		 * This is somewhat hacky regarding the output to the Results panel as the comparisons are performed twice by other code
		 * Here, a list of already-made comparisons is maintained in order to determine whether we need to output the current comparison
		 * 
		 */
		String outp=""; 
		String compare = m_Instances.attribute(comparisonColumn).name();
		
		if (!prevDataset.equals(datasetName)) {
			if (latexOutput) {
				if (!prevDataset.equals("")) { // this finishes off the previous table definition - no need to do this first time round
					outp+="\\hline\n" +  
							"\\end{tabular} \\footnotesize \\par}\n" + 
							"\\end{table}\n";
				}
					
				outp+="\n\\begin{table}[thb]\n" + 
						"\\caption{\\label{wilcoxon-"+datasetName+"} Wilcoxon test: evaluating for dataset "+datasetName
						+" based on " + compare.replace("_", "\\_") + " for Algorithm ("+(resultset1Index+1)+"), $\\alpha$ = "+m_SignificanceLevel+"\\\\}\n" + 
						"\\footnotesize\n" + 
						"{\\centering \\begin{tabular}{c||c|c|c|c|c}\n" + 
						"\\hline\n"
						+ "Comparison & R+ & R- & Exact p & Asymptotic p & Adjusted p \\\\"
						+ "\n\\hline\n";	
			}
			else {
				outp = "\nDataset: "+datasetName+", "+ "based on "+m_Instances.attribute(comparisonColumn).name()+"\n"
					+ "Comparison\tR+\t\tR-\t\tExact p\t\tAsymptotic p\tAdjusted p\n"
					+ "---------------------------------------------------------------------------------------------\n";
			}
		}
		
		prevDataset = datasetName;
		String thisComparison=datasetName+resultset1Index+resultset2Index+" "; // record this comparison
		int numAlgorithms = m_Resultsets.size();

		exact=-2;

		// perform the test
		outp += performTest(Wplus,Wminus,N,latexOutput,resultset1Index,resultset2Index,numAlgorithms);

		// use this to determine significance of results
		double pValue;

		// If we have the exact p value, use this, otherwise use the adjusted p value
		if (exact==-2) pValue=adjusted;
		else pValue=Math.abs(exact); // -1 is returned for a p value >= 0.2

		pairedStats.differencesProbability=pValue;
		pairedStats.calculateDerived();


		// -----------------------------------------
		// check the other way

		exact = -2;

		outp += performTest(Wminus,Wplus,N,latexOutput,resultset2Index,resultset1Index,numAlgorithms);

		double pValue2;
		
		// If we have the exact p value, use this, otherwise use the adjusted p value
		if (exact==-2) pValue2=adjusted;
		else pValue2=Math.abs(exact); // -1 is returned for a p value >= 0.2

		// used in the results table to denote significant differences
		if (pValue<=m_SignificanceLevel) pairedStats.differencesSignificance=1;
		else if (pValue2<=m_SignificanceLevel) pairedStats.differencesSignificance=-1;
		else pairedStats.differencesSignificance=0;
		

		// if this comparison has not already been performed we can add to the output
		if (alreadyCompared.indexOf(thisComparison)==-1) 
			output.append(outp);
		
		alreadyCompared+=thisComparison; // add to list of comparisons

		return pairedStats;
	}
	
	// Hack to prevent double output in the results panel
	private String alreadyCompared="";
	
	public void reset() {
		prevDataset = "";
		alreadyCompared="";
		output = new StringBuffer();
	}


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

		double[] z = new double[numberOfDatasets]; // differences
		double[] zAbs = new double[z.length]; // absolute differences
		int numZeros = 0;
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
					"\\caption{\\label{wilcoxon-all} Wilcoxon: evaluating across all datasets for Algorithm ("+(baseResultset+1)+"), based on " 
					+ compare.replace("_", "\\_") + ", $\\alpha$ = "+m_SignificanceLevel+"\\\\}\n" + 
					"\\footnotesize\n" + 
					"{\\centering \\begin{tabular}{c||c|c|c|c|c}\n" + 
					"\\hline\n"
					+ "Comparison & R+ & R- & Exact p & Asymptotic p & Adjusted p \\\\"
					+ "\n\\hline\n";	
		}
		else {
			outp="\n-------------------------------------------------------------------------------\n"
					+ "Wilcoxon: evaluating across all datasets\n"
					+ "-------------------------------------------------------------------------------\n\nAlgorithms:\n";
			for (int i=0;i<numAlgorithms;i++) {
				resultset = (Resultset) m_Resultsets.get(i);
				outp+="("+(i+1)+") "+resultset.templateString()+"\n";
			}

			outp += "\nFull comparison of \n("+(baseResultset+1)+") "+baseResultSet.templateString()+"\nwith all other algorithms, ";
			outp += "based on: "+compare+ ", alpha = "+m_SignificanceLevel+"\n\n";

			outp+= "Comparison\tR+\t\tR-\t\tExact p\t\tAsymptotic p\tAdjusted p\n";
			outp+="---------------------------------------------------------------------------------------------\n";

		}

		result.append(outp);

		// for each algorithm, compare with baseResultset
		for (int alg=0; alg<numAlgorithms; alg++) {
			if (alg==baseResultset) continue;

			numZeros = 0;

			for (int d=0;d<numberOfDatasets;d++) {
				z[d] = data[d][alg]-data[d][baseResultset];
				if (z[d]==0) numZeros++;
				zAbs[d] = Math.abs(z[d]);
			}


			double[] ranks = rank(zAbs); 

			double Wplus = 0;
			double Wminus = 0;

			// The following code deals with zeros
			// These are split between the two algorithms (via Wplus and Wminus)
			// If there is an odd number of zeros, then one zero is ignored.
			boolean flip = true;
			boolean oddNumberOfZeros = (numZeros%2!=0);
			boolean ignoreFirst = oddNumberOfZeros;

			for (int i = 0; i < z.length; ++i) {
				if (z[i] > 0) Wplus += ranks[i];
				else if (z[i] < 0) Wminus += ranks[i];
				else { // zero difference
					if (ignoreFirst) {	// if odd number of zeros, then ignore one result		
						ignoreFirst = false;
					}
					else {
						if (flip) Wplus += ranks[i];
						else Wminus += ranks[i];

						flip = !flip;
					}
				}
			}

			int N = z.length;
			if (oddNumberOfZeros) N -= 1;

			String summary="";
			// perform the test
			summary += performTest(Wplus,Wminus,N,latexOutput,baseResultset,alg,numAlgorithms);

			// Check the opposite way
			summary += performTest(Wminus,Wplus,N,latexOutput,alg,baseResultset,numAlgorithms);

			result.append(summary);
		}

		if (latexOutput) {
			result.append("\\hline\n" +  
					"\\end{tabular} \\footnotesize \\par}\n" + 
					"\\end{table}");
		}
		else result.append("---------------------------------------------------------------------------------------------\n\n");

		output = result;
		return "";
	}

	private double asymptotic;
	private double exact;
	private double adjusted;

	/**
	 * Performs the Wilcoxon signed ranks test. Also sets the value of private variables: asymptotic, exact, and adjusted.
	 * @param Rplus R+
	 * @param Rminus R-
	 * @param N the number of datasets
	 * @param latexOutput whether to output some of the information in LaTeX format
	 * @param baseResultset algorithm 1
	 * @param alg algorithm 2
	 * @param numAlgorithms the number of algorithms
	 * @return summary of the test results
	 */
	private String performTest(double Rplus, double Rminus, int N, boolean latexOutput, int baseResultset, int alg, int numAlgorithms) {
		String summary="";

		if (latexOutput) {
			summary = "("+(baseResultset+1)+") vs ("+(alg+1)+")" + " & "+Rplus+" & "+Rminus+" & ";
		}
		else {
			summary = "("+(baseResultset+1)+") vs ("+(alg+1)+")"
					+ "\t"+Rplus+"\t\t"+Rminus+"\t\t";
		}

		boolean significant;
		boolean exactCalculated = false;

		// Calculate exact p-value if possible	
		if (N<=Ncutoff) {
			exact = calculateExactPValue(Rminus, N, Rplus);
			exactCalculated = true;

			if (exact==-1) summary += (latexOutput? "$\\geq 0.2$ & ":">= 0.2 \t\t");
			else {
				// is this result significant?
				significant = exact<m_SignificanceLevel;

				if (latexOutput) summary += (significant? "{\\bf ":"")+String.format(decimalPlaces, exact) +(significant? "}":"")+" & ";
				else summary += String.format(decimalPlaces, exact) +(significant? "*":" ") + "\t\t";
			}
		}
		else summary+="  -  "+ (latexOutput? " & ":"\t\t");

		asymptotic = calculateAsymptoticPValue(Rminus, N);

		// Correction to reduce family-wise error rate
		adjusted = correctPValue(numAlgorithms,asymptotic);

		significant = adjusted<m_SignificanceLevel;
		significant &= !exactCalculated; // if we don't have the exact p-value, we use adjusted

		if (latexOutput) {
			summary += String.format(decimalPlaces, asymptotic)+" & "+(significant? "{\\bf ":"")+String.format(decimalPlaces, adjusted)+(significant? "}":"")+" \\\\ \n";
		}
		else summary += String.format(decimalPlaces, asymptotic)+"\t\t"+String.format(decimalPlaces, adjusted)+(significant? "*":" ")+"\n";

		return summary;
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

	// is the p-value table initialised already?
	boolean tableInitialised = false;
	double[][] exactPValues = new double[505][51];

	int Ncutoff = 50; // will be set to 20 if the table file can't be found
	boolean noTable = false; // used if there is a problem reading the table

	/**
	 * Decide which method to use for exact P values: either from a table or calculated
	 * Calculating P values takes a long time so this is limited to N <= 20
	 * @param R
	 * @param N
	 * @param otherR for use in the calculated exact P value method
	 * @return
	 */
	private double calculateExactPValue(double R, final int N, double otherR) {
		if (!tableInitialised) {
			tableInitialised=true;
			readTable();
		}

		if (noTable) return calculateExactPValue2(otherR,N);
		else return calculateExactPValue1(R,N);
	}

	/**
	 * Use a table of exact P values, stored in resources/WilcoxonTable.csv
	 * @param R
	 * @param N
	 * @return p value: -1 if the p value is >=0.2 
	 */
	private double calculateExactPValue1(double R, final int N) {
		if (N>50) return -1;
		else if (R>505) return -1;

		double value1, value2;
		value1 = exactPValues[(int)Math.ceil(R)][N];
		//if (value1==-1) value1=1;

		value2 = exactPValues[(int)Math.floor(R)][N];
		//if (value2==-1) value2=1;

		return (value1+value2)/2;
	}

	/**
	 * Algorithm inspired by
	 * http://www.fon.hum.uva.nl/Service/Statistics/Signed_Rank_Algorihms.html#C
	 * by Rob van Son, Institute of Phonetic Sciences & IFOTT,
	 * University of Amsterdam
	 *
	 * @param R Wilcoxon signed rank value
	 * @param N number of subjects (corresponding to x.length)
	 * @return two-sided exact p-value
	 */
	private double calculateExactPValue2(double R, final int N) {
		// Total number of outcomes (equal to 2^N but a lot faster)
		final int m = 1 << N;
		//double threshold = 0.1*m;

		int largerRankSums = 0;

		for (int i = 0; i < m; ++i) {
			int rankSum = 0;

			// Generate all possible rank sums
			for (int j = 0; j < N; ++j) {

				// (i >> j) & 1 extract i's j-th bit from the right
				if (((i >> j) & 1) == 1) {
					rankSum += j + 1;
				}
			}

			if (rankSum >= R) {
				++largerRankSums;
			}

			//if (largerRankSums>=threshold) return -1; // flag that this p value is >= 0.2 so we can stop calculations
		}


		/*
		 * largerRankSums / m gives the one-sided p-value, so it's multiplied
		 * with 2 to get the two-sided p-value
		 */
		return Math.min(1, 2 * ((double) largerRankSums) / ((double) m));
	}

	/**
	 * @param Wmin smallest Wilcoxon signed rank value
	 * @param N number of subjects (corresponding to x.length)
	 * @return two-sided asymptotic p-value
	 */
	private double calculateAsymptoticPValue(final double Wmin, final int N) {

		final double ES = (double) (N * (N + 1)) / 4.0;

		/* Same as (but saves computations):
		 * final double VarW = ((double) (N * (N + 1) * (2*N + 1))) / 24;
		 */
		final double VarS = ES * ((double) (2 * N + 1) / 6.0);

		// - 0.5 is a continuity correction
		final double z = (Wmin - ES - 0.5) / Math.sqrt(VarS);

		// was return 2*standardNormal.cumulativeProbability(z);
		return Math.min(1,2*Statistics.normalProbability(z));
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
	 * Read in the table of exact p values from resources folder
	 */
	private void readTable() {
		try {
			//URL path = WilcoxonSignedRank.class.getResource("WilcoxonTable.csv");
			//File myObj = new File(path.getFile());
			//Scanner myReader = new Scanner(myObj);

			InputStream path = WilcoxonSignedRank.class.getResourceAsStream("WilcoxonTable.csv");
			Scanner myReader = new Scanner(path);

			//File myObj = new File("WilcoxonTable.csv");

			for(int i=0;i<exactPValues.length;i++){
				Arrays.fill(exactPValues[i],-1.0);
			}

			int r = 0; // row index

			while (myReader.hasNextLine()) {
				String row = myReader.nextLine().trim();
				StringTokenizer tokenizer = new StringTokenizer(row, ",");

				int n = 4;
				while (tokenizer.hasMoreTokens()) {
					exactPValues[r][n] = Double.valueOf(tokenizer.nextToken().trim());
					n++;
				}

				r++;

			}
			myReader.close();
		}
		catch (Exception e) {
			System.err.println(e);
			System.err.println("Will calculate exact P values rather than read from table (N<=20)");
			noTable = true;
			Ncutoff = 20;
		}
	}

	/**
	 * Test the class from the command line.
	 * 
	 * @param args contains options for the instance ttests
	 */
	public static void main(String args[]) {

		try {
			// data from: https://sci2s.ugr.es/sites/default/files/files/TutorialsAndPlenaryTalks/INIT-AERFAI-Course-Statistical_Analysis_of_Experiments.pdf
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

			WilcoxonSignedRank tt = new WilcoxonSignedRank();

			//comparison columns
			int comp1=0;
			int comp2=1;
			int numZeros=0;

			double[] zAbs = new double[data.length];
			double[] z = new double[data.length];
			for (int i=0;i<zAbs.length;i++) {
				z[i] = data[i][comp1]-data[i][comp2];
				zAbs[i] = Math.abs(z[i]);
			}

			double[] ranks = tt.rank(zAbs); 

			//System.err.println(Arrays.toString(z)); // print out the differences
			System.err.println("Ranks: "); // print out the ranks
			System.err.println(Arrays.toString(ranks));

			double Wplus = 0;
			double Wminus = 0;
			boolean flip = true;
			boolean oddNumberOfZeros = (numZeros%2!=0);
			boolean ignoreFirst = oddNumberOfZeros;

			for (int i = 0; i < z.length; ++i) {
				if (z[i] > 0) Wplus += ranks[i];
				else if (z[i] < 0) Wminus += ranks[i];
				else { // zero difference
					if (ignoreFirst) {				
						ignoreFirst = false;
					}
					else {
						if (flip) Wplus += ranks[i];
						else Wminus += ranks[i];

						flip = !flip;
					}
				}
			}

			int N = z.length;
			if (oddNumberOfZeros) N -= 1;

			String summary = "\nAlgorithms\tR+\t\tR-\t\tExact p\t\tAsymptotic p\n"
					+ "("+(comp1+1)+") and ("+(comp2+1)+")" 
					+ "\t"+Wplus+"\t\t"+Wminus+"\t\t";

			// Exact P-value	
			if (N<=50) {
				double exact = tt.calculateExactPValue(Wminus, N, Wplus);
				if (exact==-1) summary += ">=0.2\t\t";
				else summary += String.format(decimalPlaces, exact)+"\t\t";
			}
			else summary+="---\t\t";

			double asymptotic =  tt.calculateAsymptoticPValue(Wminus, N);

			summary += String.format(decimalPlaces, asymptotic)+"\n";

			// compare the other way...
			double temp = Wminus;
			Wminus=Wplus; Wplus=temp;

			summary += "("+(comp2+1)+") and ("+(comp1+1)+")" 
					+ "\t"+Wplus+"\t\t"+Wminus+"\t\t";
			// Exact P-value	
			if (N<=50) {
				double exact = tt.calculateExactPValue(Wminus, N, Wplus);
				if (exact==-1) summary += ">=0.2\t\t";
				else summary += String.format(decimalPlaces, exact)+"\t\t";
			}
			else summary+="---\t\t";

			asymptotic =  tt.calculateAsymptoticPValue(Wminus, N);

			summary += String.format(decimalPlaces, asymptotic)+"\n";


			System.out.println(summary);

			// Another test
			Wplus = 385;
			Wminus = 210;
			N = 34;
			System.out.println("Exact: "+tt.calculateExactPValue(Wminus,N,Wplus));// should be 0.13832
			System.out.println(tt.calculateAsymptoticPValue(Wminus, N)); // should be 0.132454





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
		return "Wilcoxon Signed Rank Test";
	}

	/**
	 * returns a string that is displayed as tooltip on the "perform test" button
	 * in the experimenter
	 * 
	 * @return the string for the tool tip
	 */
	@Override
	public String getToolTipText() {
		return "Performs test using Wilcoxon Signed Rank Test";
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

	/**
	 * Represents the position of a double value in an ordering.
	 * Comparable interface is implemented so Arrays.sort can be used
	 * to sort an array of IntDoublePairs by value.  Note that the
	 * implicitly defined natural ordering is NOT consistent with equals.
	 */
	public static class IntDoublePair implements Comparable<IntDoublePair>  {

		/** Value of the pair */
		private final double value;

		/** Original position of the pair */
		private final int position;

		/**
		 * Construct an IntDoublePair with the given value and position.
		 * @param value the value of the pair
		 * @param position the original position
		 */
		IntDoublePair(double value, int position) {
			this.value = value;
			this.position = position;
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

		/**
		 * Returns the original position of the pair.
		 * @return position
		 */
		public int getPosition() {
			return position;
		}
	}
}
