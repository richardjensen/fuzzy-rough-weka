package weka.fuzzy.tnorm;

import java.util.Enumeration;
import java.util.Vector;
import weka.core.*;
import weka.fuzzy.snorm.SNorm;

import java.io.Serializable;


public abstract class TNorm implements OptionHandler, Serializable {
	
	public TNorm(){
		
	}
	
	public abstract double calculate(double a, double b);


	public String[] getOptions() {
		Vector<String>	result;
	    
	    result = new Vector<String>();
	    return result.toArray(new String[result.size()]);
	}

	public Enumeration listOptions() {
		Vector result = new Vector();
		return result.elements();
	}

	public void setOptions(String[] options) throws Exception {
		// no options
		
	}
    public abstract String toString();
    
    public abstract String globalInfo();
    
    public abstract SNorm getAssociatedSNorm();
}
