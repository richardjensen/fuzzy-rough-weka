package weka.fuzzy.implicator;

public class ImplicatorKD extends Implicator {
	/** for serialization. */
	private static final long serialVersionUID = 1065606253458807303L;
	
	public ImplicatorKD(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.max(1 - a, b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "The Kleene-Dienes implicator:\n"
			 + "I(x,y) = max(1 - x, y).";
		
	}
	
	public String toString() {
		return "Kleene-Dienes";
	}
}
