package weka.classifiers.fuzzy;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.Option;
import weka.fuzzy.tnorm.*;
import weka.fuzzy.snorm.*;
import weka.fuzzy.similarity.*;

import java.io.Serializable;
import java.util.*;


public abstract class FuzzyRoughRuleInducer extends AbstractClassifier implements Serializable {
	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	/** The training instances used for classification. */
	protected Instances m_Train;

	/** The number of class values (or 1 if predicting numeric). */
	protected int m_numClasses;

	/** The class attribute type. */
	protected int m_ClassType;


	public TNorm m_composition = new TNormKD();
	public TNorm m_TNorm = new TNormLukasiewicz();
	public Similarity m_Similarity = new Similarity1();
	public int m_numInstances;

	//public Similarity m_DecisionSimilarity = new SimilarityEq();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public SNorm m_SNorm;
	public int m_classIndex;
	public double[] globallowers;
	public double[] global;

	public boolean fullData=false;
	public Relation current;
	public double[] vals;
	public ArrayList<FuzzyRule> cands;
	public double arity=0;
	public Relation temprel;
	public ArrayList<FuzzyRule> cands2;
	public int[] decIndexes;
	public double[] decVals;
	public int[] indexes;
	public int m_Pruning=0;
	
	public class FuzzyRule implements Serializable {
		double[] values;
		BitSet attributes;
		int object;

		/** for serialization. */
		static final long serialVersionUID = -3080186098777067142L;

		public FuzzyRule(double[] v, BitSet a, int obj) {
			values=v;
			attributes=a;
			object=obj;
		}
		

		public int hashCode() {
			return Arrays.hashCode(values);
		}

		public String toString() {
			String ret="Rule - \n "+attributes.toString()+":\n Object: "+object+"\n Eq class: [ ";
			//for (int i=0;i<values.length;i++) ret+=values[i]+" ";
			
			ret+="]";
			return ret;
		}
	}
	public FuzzyRoughRuleInducer() {
		
	}
	
	public abstract void induce(double full);
	
	
	public final void initCurrent() {
		for (int i=0;i<m_numInstances;i++) {
			for (int j=i;j<m_numInstances;j++) {
				current.setCell(i,j,1);
			}
		}
	}

	public final void generatePartition(Relation curr, int a) {
		double mainValue;

		for (int o=0;o<m_numInstances;o++) {
			temprel.setCell(o,o,1);
			mainValue = m_Train.instance(o).value(a);

			for (int o1=o+1;o1<m_numInstances;o1++) {
				double rel =  m_composition.calculate(fuzzySimilarity(a, mainValue, m_Train.instance(o1).value(a)),curr.getCell(o,o1));
				temprel.setCell(o,o1,rel);
			}
		}
	}


	public final void setCurrent(Relation rel) {
		for (int i=0;i<m_numInstances;i++) {
			for (int j=i;j<m_numInstances;j++) {
				current.setCell(i,j,rel.getCell(i,j));
			}
		}
	}


	public final void setIndexes(Relation rel) {
		int m_numInstances = rel.size;

		for (int i = 0; i < m_numInstances; i++) {
			boolean same = false;

			for (int j = i + 1; j < m_numInstances; j++) {

				for (int d = 0; d < m_numInstances; d++) {
					if (rel.getCell(i, d) != rel.getCell(j, d)) {
						same = false;
						break;
					} else
						same = true;
				}
				if (same) {
					indexes[j] = i + 1;
					break;
				} // i+1 as default is 0
			}

		}
	}

	

	public void determineAdd(FuzzyRule temp) {
		boolean[] removes =  new boolean[cands.size()];
		int index=0;
		boolean superset=false;
		boolean isSubset=false;
		FuzzyRule list1=null;
		int siz=0;

		Iterator<FuzzyRule> it = cands.iterator();
		index=0;
		while(it.hasNext()) {
			list1 = it.next();
			siz = compare(temp,list1);

			//already covered by an existing rule
			if (siz==1) {
				isSubset=true;   
				break;
			}
			//else if superset
			else if (siz==-1) {
				superset=true;
				removes[index]=true;    
				//can delete this next line to remove other supersets
				//break;
			}
			index++;
		}


		//the rule 'temp' can replace others in the ruleset, so remove these
		if (superset) {
			cands2 = new ArrayList<FuzzyRule>(removes.length);
			it = cands.iterator();
			index=0;
			while(it.hasNext()) {
				list1 = it.next();
				if (!removes[index]) {
					cands2.add(list1);
				}
				index++;
			}
			cands2.add(temp);

			updateCoverage(temp);

			cands=cands2;
		}
		else if (!isSubset)  { //haven't encountered this rule before
			cands.add(temp);
			updateCoverage(temp);
		} 
	}

	private void updateCoverage(FuzzyRule rp) {
		for (int o=0;o<m_numInstances;o++) {
			global[o] = m_SNorm.calculate(rp.values[o],global[o]);
		}

	}
	
	public final boolean allCovered() {
		boolean stop=true;
		
		for (int x=0;x<m_numInstances;x++) {
			if ((float)global[x]!=(float)globallowers[x]) {stop=false; break;}
		}
		
		return stop;
	}

	//	compares two ruleparts.
	//	if r1 is a subset= of r2, return 1
	//	if r2 is a subset of r1, return -1;
	//	else return 0
	private final int compare(FuzzyRule r1, FuzzyRule r2) {
		double sum=0;
		double r1sum=0;
		double r2sum=0;

		for (int o=0;o<m_numInstances;o++) {
			//result[o] = Math.min(r1.values[o],r2.values[o]);
			sum+=m_TNorm.calculate(r1.values[o],r2.values[o]);
			r1sum+=r1.values[o];
			r2sum+=r2.values[o];
			
			//if (sum!=r1sum && sum!=r2sum) return 0;
		}

		if (sum==r1sum) {  return 1;}  
		else if (sum==r2sum) { return -1;}
		else return 0;
	}

	public final double fuzzySimilarity(int attr, double x, double y) {
		double ret = 0;			

		//no decision feature, so each object is distinct
		if (attr<0 && attr==m_classIndex) {
			ret=0;
		}
		else {
			//if it's a nominal attribute, then use crisp equivalence
			//otherwise use the general similarity measure
			if (Double.isNaN(x)||Double.isNaN(y)) ret=1;	
			else if (m_Train.attribute(attr).isNominal()) ret = m_SimilarityEq.similarity(attr, x, y);
			else ret = m_Similarity.similarity(attr, x, y);

		}
		return ret;
	}
	
	
	
	public void prune() {
		Iterator<FuzzyRule> it = cands.iterator();
		FuzzyRule rule;

		while (it.hasNext()) {
			rule = it.next();
			
			for (int a = rule.attributes.nextSetBit(0); a >= 0; a = rule.attributes.nextSetBit(a + 1)) {
				rule.attributes.clear(a);
				boolean remove=true;
				
				for (int o=0;o<m_numInstances;o++) {
					double value=1;
					
					for (int b = rule.attributes.nextSetBit(0); b >= 0; b = rule.attributes.nextSetBit(b + 1)) {
						value = m_composition.calculate(fuzzySimilarity(b,m_Train.instance(o).value(b),m_Train.instance(rule.object).value(b)), value);
					}
					if ((float)value!=(float)rule.values[o]) {
						remove=false;
						break;
					}
				}
				if (!remove) rule.attributes.set(a); 
			}
		}
	}
	
	public abstract Enumeration<Option> listOptions();
}
