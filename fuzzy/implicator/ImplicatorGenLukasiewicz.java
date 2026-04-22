package weka.fuzzy.implicator;

import java.util.Vector;

import weka.core.Utils;

//Generalised operator
public class ImplicatorGenLukasiewicz extends Implicator {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807303L;
	private double alpha=1;
	
	public ImplicatorGenLukasiewicz(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.pow(Math.min(1, 1- Math.pow(a,alpha) + Math.pow(b,alpha)),1/alpha);
	} 

	/**
	 * Gets the current settings. 
	 *
	 * @return 		an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions(){
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-A");
		result.add("" + getAlpha());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		String tmpStr = Utils.getOption('A', options);
		if (tmpStr.length() != 0)
			setAlpha(Double.parseDouble(tmpStr));
		else
			setAlpha(1.0);

	}

	public void setAlpha(double a) {
		alpha=a;
	}

	public double getAlpha() {
		return alpha;
	}

	
	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "The Lukasiewicz implicator:\n"
			 + "I(x,y) = min(1, 1 - x + y).";
		
	}
	public String toString() {
		return "Lukasiewicz";
	}
}
