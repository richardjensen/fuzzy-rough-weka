package weka.fuzzy.snorm;

public class SNormKD extends SNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public SNormKD(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.max(a,b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "Kleene-Dienes S-norm: S(x,y) = max(x,y).";
	}
	
	public String toString() {
		return "Kleene-Dienes";
	}
}
