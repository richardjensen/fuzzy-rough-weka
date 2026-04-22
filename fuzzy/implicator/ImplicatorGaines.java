package weka.fuzzy.implicator;

public class ImplicatorGaines extends Implicator {
	/** for serialization. */
	private static final long serialVersionUID = 1065606253458807303L;
	
	public ImplicatorGaines(){
	  super();	
	}
	
	public double calculate(double a, double b) {
		if (a <= b) {
			return 1;
		} else {
			return b / a;
		}
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "The Gaines implicator:\n"
		+"I(x,y)=1 if x <= y, else I(x,y)=y/x";
		
	}
	public String toString() {
		return "Gaines";
	}
	
}
