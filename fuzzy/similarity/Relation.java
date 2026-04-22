package weka.fuzzy.similarity;
import java.io.Serializable;

//Note that this class assumes that the relation is symmetric in order to save memory use
public class Relation implements Serializable {
	 /** for serialization */
	  static final long serialVersionUID = 747878400815262784L;
	  
        public int size=0;
        double[][] vals;

        public Relation() {}
        public Relation(int s) {
                size=s;
                //vals = new double[s*s];
                //vals = new double[s][s];
                
                vals = new double[s][];
                
                for (int i = 0; i < s; i++) {
          	      vals[i] = new double [i+1];
          	    }
                
        }

        public void setCell(int i, int j, double value) {
                //vals[i*size+j] = value;
        		if (i>=j) vals[i][j]=value; 
        		else vals[j][i]=value;
        	//vals[i][j] = value;
        }

        public double getCell(int i, int j) {
                //return vals[i*size+j];
        	if (i>=j) return vals[i][j];
        	else return vals[j][i];
        	//return vals[i][j];
        }


}//end of class