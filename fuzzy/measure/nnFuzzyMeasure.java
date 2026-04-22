package weka.fuzzy.measure;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;
import weka.core.Instances;
import weka.core.Utils;


public abstract class nnFuzzyMeasure extends FuzzyMeasure {
	public int knn=1;
	
	public int[][] neighbours;

	public class Index implements Comparable<Index>{
		int object;
		double sim=0;

		public Index(int a, double g) {
			object = a;
			sim = g;
		}

		public int object() {
			return object;
		}

		//sort in descending order (best first)
		public int compareTo(Index o) {
			return Double.compare(o.sim,sim);
		}

		public String toString() {
			return object+":"+sim;
		}

	}
	
	/**
	 * Gets the current settings. Returns empty array.
	 *
	 * @return 		an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions(){
		Vector<String>	result = new Vector<String>();

		result.add("-K");
		result.add("" + getKnn());

		return result.toArray(new String[result.size()]);
	}
	
	public int getKnn() {
		return knn;
	}

	public void setKnn(int k) {
		knn=k;
	}
	
	/**
	 * Parses a given list of options.
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		String	tmpStr;

		tmpStr = Utils.getOption('K', options);
		if (tmpStr.length() != 0)
			setKnn(Integer.parseInt(tmpStr));
		else
			setKnn(0);

	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1234234L;

	//Find the neighbours of each data instance
	public void initialiseNeighbours(Instances m_Train) {
		
		//make sure that the k value is within range
		if (getKnn()>m_numInstances-1) {
			setKnn(m_numInstances-1);
		}
		
		neighbours = new int[m_Train.numInstances()][getKnn()];
		//Similarity condSim = new Similarity1();
		//condSim.setInstances(m_Train);
		
		for(int x=0; x<m_Train.numInstances(); x++) {
			ArrayList<Index> objectIndex = new ArrayList<Index>(m_Train.numInstances()); //keep track of neighbours (this list is sorted later)
			double classValue = m_Train.instance(x).classValue(); //the class value of the current instance, x.

			for(int o=0; o<m_Train.numInstances(); o++) {
				if (o!=x && m_Train.instance(o).classValue()!=classValue) {
					double sim=1;

					//compute similarity using all features
					for(int i = 0; i< m_Train.numAttributes(); i++){
						if (i != m_Train.classIndex()) {
							double similarity = fuzzySimilarity(i, x, o);
							sim+=similarity*similarity;
							//sim = m_composition.calculate(sim, similarity);
						}
					}
					objectIndex.add(new Index(o, sim));
				}
			}

			//sort the neighbours
			Collections.sort(objectIndex);

			//find the k-nearest neighbours
			for (int o=0;o<getKnn();o++) {
				neighbours[x][o] = objectIndex.get(o).object();
			}
		}
		

	}
	

}
