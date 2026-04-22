
package weka.fuzzy.tnorm;

import java.util.Vector;

import weka.core.Utils;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormGenLukasiewicz;

// Generalised operator
public class TNormGenLukasiewicz extends TNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	private double alpha = 1; //alpha=1 makes this behave like the standard Lukasiewicz operator
	
	public TNormGenLukasiewicz(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.pow(Math.max(0, Math.pow(a,alpha) + Math.pow(b,alpha)-1),1/alpha);
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
		return "Lukasiewicz T-norm: T(x,y) = max(x+y-1,0).";		
	}
	
	public String toString() {
		return "Lukasiewicz";
	}
	
	public SNorm getAssociatedSNorm() {
		return new SNormGenLukasiewicz(alpha);
	}
}
