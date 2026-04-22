package weka.fuzzy.snorm;

import java.util.Enumeration;
import java.util.Vector;
import weka.core.*;
import java.io.Serializable;


public abstract class SNorm implements OptionHandler, Serializable {
	
	public SNorm(){
		
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
}
