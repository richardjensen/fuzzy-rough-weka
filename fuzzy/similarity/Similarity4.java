/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    EuclideanDistance.java
 *    Copyright (C) 1999-2007 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.fuzzy.similarity;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.*;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormKD;

/**
 <!-- globalinfo-start -->
 *  Similarity measure: max(0, min(1, beta - alpha * abs(a(x) - a(y)) / l(a).
 *
 * 
 <!-- globalinfo-end -->
 *

 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  Turns off the normalization of attribute 
 *  values in distance calculation.</pre>
 * 
 * <pre> -R &lt;col1,col2-col4,...&gt;
 *  Specifies list of columns to used in the calculation of the 
 *  distance. 'first' and 'last' are valid indices.
 *  (default: first-last)</pre>
 * 
 * <pre> -V
 *  Invert matching sense of column indices.</pre>
 *  
 * <pre> -T
 *  determines the t-norm to use </pre>
 * <pre> -A
 *  set alpha </pre>
 * <pre> -B
 *  set beta </pre>
 * <pre> -C
 *  set cutoff value </pre>
 * 
 <!-- options-end --> 
 *
 */
public class Similarity4
extends Similarity
implements Cloneable, TechnicalInformationHandler {

	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;

	private double alpha = 0.5;
	private double beta = 1.0;



	/**
	 * Constructs Similarity based Distance object, Instances must be still set.
	 */
	public Similarity4() {
		super();
	}

	/**
	 * Constructs an Similarity based Distance object and automatically initializes the
	 * ranges.
	 * 
	 * @param data 	the instances the distance function should work on
	 */
	public Similarity4(Instances data) {
		super(data);
	}

	/**
	 * Returns a string describing this object.
	 * 
	 * @return 		a description of the evaluator suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return 
		"The similarity measure is: \n" 
		+ "max(0, min(1, beta - alpha * abs(a(x) - a(y)) / l(a).\n";
	}

	public String toString() {
		return "max(0, min(1, "+beta+" - "+alpha+" * abs(a(x) - a(y)) / l(a).";
	}
	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return 		the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.MISC);
		//to be filled in
		return result;
	}

	@Override

	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.add(new Option(
				"\tTurns off the normalization of attribute \n"
				+ "\tvalues in distance calculation.",
				"D", 0, "-D"));

		result.addElement(new Option(
				"\tSpecifies list of columns to used in the calculation of the \n"
				+ "\tdistance. 'first' and 'last' are valid indices.\n"
				+ "\t(default: first-last)",
				"R", 1, "-R <col1,col2-col4,...>"));

		result.addElement(new Option(
				"\tInvert matching sense of column indices.",
				"V", 0, "-V"));

		result.addElement(new Option(
				"\tT-norm to use.\n"
				+ "\t(default: weka.similarity.TNormMinimum)",
				"T", 1,"-T <classname and options>"));

		result.addElement(new Option(
				"\tgranularity parameter alpha\n"
				+ "\t(default: 1.0)",
				"A", 1,"-A <value>"));


		result.addElement(new Option(
				"\tgranularity parameter beta\n"
				+ "\t(default: 1.0)",
				"B", 1,"-B <value>"));

		result.addElement(new Option(
				"\tcutoff value to use\n"
				+ "\t(default: 0.0)",
				"C", 1,"-C <value>"));


		return result.elements();
	}

	/**
	 * Gets the current settings. Returns empty array.
	 *
	 * @return 		an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions(){
		Vector<String>	result;

		result = new Vector<String>();

		if (getDontNormalize())
			result.add("-D");

		result.add("-R");
		result.add(getAttributeIndices());

		if (getInvertSelection())
			result.add("-V");

		result.add("-T");
		result.add((m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim());

		result.add("-A");
		result.add("" + getAlpha());

		result.add("-B");
		result.add("" + getBeta());

		result.add("-C");
		result.add("" + getCutoff());


		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		String	tmpStr;

		setDontNormalize(Utils.getFlag('D', options));

		tmpStr = Utils.getOption('R', options);
		if (tmpStr.length() != 0)
			setAttributeIndices(tmpStr);
		else
			setAttributeIndices("first-last");

		setInvertSelection(Utils.getFlag('V', options));

		String nnSearchClass = Utils.getOption('T', options);
		if(nnSearchClass.length() != 0) {
			String moreOptions[] = Utils.splitOptions(nnSearchClass);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid TNorm specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setTNorm( (TNorm) Utils.forName( TNorm.class, className, moreOptions) );
		}
		else {
			setTNorm(new TNormKD());
		}

		tmpStr = Utils.getOption('A', options);
		if (tmpStr.length() != 0)
			setAlpha(Double.parseDouble(tmpStr));
		else
			setAlpha(1.0);

		tmpStr = Utils.getOption('B', options);
		if (tmpStr.length() != 0)
			setBeta(Double.parseDouble(tmpStr));
		else
			setBeta(1.0);

		tmpStr = Utils.getOption('C', options);
		if (tmpStr.length() != 0)
			setCutoff(Double.parseDouble(tmpStr));
		else
			setCutoff(0.0);

	}


	public String alphaTipText() {
		return "The granularity parameter alpha " +
		"(default: 1.0). ";
	}


	public double getAlpha() {
		return alpha;
	}


	public void setAlpha(double a) throws Exception {
		alpha = a;
	}

	public String betaTipText() {
		return "The granularity parameter beta " +
		"(default: 1.0). ";
	}


	public double getBeta() {
		return beta;
	}


	public void setBeta(double b) throws Exception {
		beta = b;
	}


	/**
	 * similarity function calculates similarity between 2 attribute values
	 * 
	 * @param index	the index of the attribute for which the similarity is calculated
	 * @param first 	the value of the attribute of the first instance
	 * @param second 	the value of the attribute of the second instance
	 * 
	 * @return 	a measurement of similarity
	 * 
	 */
	public double similarity(int index, double first, double second){

		return Math.max(0 , Math.min(1, beta - alpha*Math.abs(first - second)/attrDifference[index]));
	}

	//Interval-valued version
	  public double[] similarity(int index, double first, double second, double param){
		  double[] ret = new double[2];  
		  ret[0] = param*Math.max(0 , Math.min(1, beta - alpha*Math.abs(first - second)/attrDifference[index]));
		  ret[1] = Math.max(0 , Math.min(1, beta - alpha*Math.abs(first - second)/attrDifference[index]));
		  return ret;
	  }

	@Override
	public void clean() {
		// TODO Auto-generated method stub
		
	}
}
