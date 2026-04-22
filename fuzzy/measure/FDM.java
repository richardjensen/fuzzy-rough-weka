package weka.fuzzy.measure;

import java.util.BitSet;
import java.util.Vector;
import java.util.Iterator;
import weka.core.Utils;

public class FDM extends FuzzyDiscernibilityMeasure {


	private static final long serialVersionUID = 1063606753458807303L;
	
	//Determine whether to use the sum or tnorm
	public boolean sum=true;
	
	public FDM() {
		super();
	}
	
	/**
	 * Search through the clauses and evaluate the reduct
	 */
	public double calculate(BitSet reduct) {
		double ret;
		
		if (sum) ret=0;
		else ret=1;
		fdm_length = clauseList.size();
		Iterator<Clause> e;
		Clause cl;
		
		for (e = clauseList.iterator(); e.hasNext(); ) {
			double current=0;
			cl = e.next();
			for (int a = reduct.nextSetBit(0); a >= 0; a = reduct.nextSetBit(a + 1)) {
				if (a!=m_classIndex) current = m_composition.getAssociatedSNorm().calculate(current, cl.getVariableValue(a));
			}
			
			if (sum) ret+= m_Implicator.calculate(cl.getVariableValue(m_classIndex), current);
			else {
				ret = m_TNorm.calculate(ret, m_Implicator.calculate(cl.getVariableValue(m_classIndex), current));
				if (ret==0) return 0;
			}
		}
			
		if (sum) return ret/fdm_length;
		else return ret;
	}
	
	public String globalInfo() {
		return "Discernibility matrix-based search.\n\nIf 'simplify' is set to 'true' then the clause database is simplified as it is created, saving memory consumption but this will be slower.";
	}

	public String toString() {
		return "FDM";
	}
	
	/**
	 * Gets the current settings. Returns empty array.
	 *
	 * @return 		an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions(){
		Vector<String>	result;

		result = new Vector<String>();

		if (getSum()) {
			result.add("-S");
		}
		
		if (getSimplify()) {
			result.add("-L");
		}
		
		if (getDebug()) {
			result.add("-D");
		}

		if (getIgnoreDecision()) {
			result.add("-I");
		}

		return result.toArray(new String[result.size()]);
	}
	
	public void setSum(boolean s) {
		sum = s;
	}
	
	public boolean getSum() {
		return sum;
	}
	
	public void setIgnoreDecision(boolean s) {
		m_ignoreDecision = s;
	}
	
	public boolean getIgnoreDecision() {
		return m_ignoreDecision;
	}
	
	public void setSimplify(boolean s) {
		simplify = s;
	}
	
	public boolean getSimplify() {
		return simplify;
	}
	
	public void setDebug(boolean d) {
		debug = d;
	}
	
	public boolean getDebug() {
		return debug;
	}
	
	/**
	 * Parses a given list of options.
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		setSum(Utils.getFlag('S', options));
		setSimplify(Utils.getFlag('L', options));
		setDebug(Utils.getFlag('D', options));
		setIgnoreDecision(Utils.getFlag('I', options));
		
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
