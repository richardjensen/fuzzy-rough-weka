/**
 * 
 */
package weka.filters.supervised.instance;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.filters.supervised.instance.bireduct.*;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.*;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity2;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;



/** 
<!-- globalinfo-start -->
 * Filters instances according to the value of an attribute.
 * <p/>
<!-- globalinfo-end -->
 * 
<!-- options-start -->
 * Valid options are: <p/>
 * 
<!-- options-end -->
 *
 * @author ncm
 * @version $Revision: 1.0 $
 */
public class BireductFilter extends Filter
implements SupervisedFilter, OptionHandler, Serializable {

	/** for serialization */
	static final long serialVersionUID = 4752870393679263361L;

	/** Stores which values of nominal attribute are to be used for filtering.*/
	protected Range m_Values;

	/** Stores which value of a numeric attribute is to be used for filtering.*/
	protected double m_Value = 0;

	/** True if missing values should count as a match */
	protected boolean m_MatchMissingValues = false;

	/** Modify header for nominal attributes? */
	protected boolean m_ModifyHeader = false;

	/** If m_ModifyHeader, stores a mapping from old to new indexes */
	protected int [] m_NominalMapping;

	public Similarity m_Similarity = new Similarity3();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();
	public FuzzyMeasure m_Measure= new BiFDM();
	public JohnsonReducer2 reducer;// = new JohnsonReducer();
	public boolean m_objectFreq=true;

	/** holds the selected attributes  */
	private int [] m_SelectedAttributes;

	/** holds the selected objects  */
	private int []  m_ObjectsToRemove;


	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();

	public ArrayList<Clause> clauseList=null;

	public boolean iterative=false;

	/**
	 * 
	 */
	public BireductFilter() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	@Override
	public Enumeration listOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String optionString;	   

		optionString = Utils.getOption('Z', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid FuzzyMeasure specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setMeasure( (FuzzyMeasure) Utils.forName( FuzzyMeasure.class, className, moreOptions) );
		}
		else {
			setMeasure(new WeakGamma());
		}


		optionString = Utils.getOption('I', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Implicator specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setImplicator( (Implicator) Utils.forName( Implicator.class, className, moreOptions) );
		}
		else {
			setImplicator(new ImplicatorLukasiewicz());
		}


		optionString = Utils.getOption('T', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid TNorm specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setTNorm( (TNorm) Utils.forName( TNorm.class, className, moreOptions) );
		}
		else {
			setTNorm(new TNormLukasiewicz());
		}

		optionString = Utils.getOption('R', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Similarity specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setSimilarity( (Similarity) Utils.forName( Similarity.class, className, moreOptions) );
		}
		else {
			setSimilarity(new Similarity2());
		}

		setObjectFreq(Utils.getFlag('F', options));
	}




	public void setMeasure(FuzzyMeasure fe) {
		m_Measure = fe;
	}

	public FuzzyMeasure getMeasure() {
		return m_Measure;
	}


	public void setImplicator(Implicator impl) {
		m_Implicator = impl;
	}

	public Implicator getImplicator() {
		return m_Implicator;
	}

	//set the relation composition operator = tnorm
	public void setTNorm(TNorm tnorm) {
		m_TNorm = tnorm;
		//m_composition = tnorm;
		m_SNorm = tnorm.getAssociatedSNorm();
	}

	public TNorm getTNorm() {
		return m_TNorm;
	}

	public Similarity getSimilarity() {
		return m_Similarity;
	}

	public void setSimilarity(Similarity s) {
		m_Similarity = s;
	}

    public boolean getObjectFreq(){
    	return m_objectFreq;
    }
    
    public void setObjectFreq(boolean freq){
    m_objectFreq = freq;	
    }

	public boolean getIterative() {
		return iterative;
	}

	public void setIterative(boolean b) {
		iterative=b;
	}

	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-Z");
		result.add((m_Measure.getClass().getName() + " " +
				Utils.joinOptions(m_Measure.getOptions())).trim());

		result.add("-I");
		result.add((m_Implicator.getClass().getName() + " " +
				Utils.joinOptions(m_Implicator.getOptions())).trim());

		result.add("-T");
		result.add((m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim());

		result.add("-R");
		result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());

		if(getObjectFreq()){
			result.add("-F");
		}
		
		if (getIterative()) {
			result.add("-B");
		}

		return result.toArray(new String[result.size()]);
	}

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// attributes
		result.enableAllAttributes();

		// class
		result.enableAllClasses();
		result.enable(Capability.NO_CLASS);

		return result;
	}	


	protected void setOutputFormat() throws Exception {
		Instances informat;

		if (m_SelectedAttributes == null) {
			setOutputFormat(null);
			return;
		}

		//System.out.println("reduct length = " + m_SelectedAttributes.length);

		ArrayList<Attribute> attributes = new ArrayList<Attribute>(m_SelectedAttributes.length);

		informat = getInputFormat();

		for (int i=0;i < m_SelectedAttributes.length;i++) {
			attributes.add((Attribute)informat.attribute(m_SelectedAttributes[i]).copy());
			//System.err.println(attributes.get(i));
		}

		Instances outputFormat = 
				new Instances(getInputFormat().relationName(), attributes, 0);

		//System.out.println("classIndex is: " + (m_SelectedAttributes.length-1));
		outputFormat.setClassIndex(m_SelectedAttributes.length-1);

		setOutputFormat(outputFormat);  
	}

	public boolean input(Instance instance) throws Exception {

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		if (m_NewBatch) {
			resetQueue();
			m_NewBatch = false;
		}

		if (isOutputFormatDefined()) {
			convertInstance(instance);
			return true;
		}

		bufferInput(instance);
		return false;
	}

	/**
	 * Convert a single instance. Selected attributes only are transfered.
	 * The converted instance is added to the end of
	 * the output queue.
	 *
	 * @param instance the instance to convert
	 * @throws Exception if something goes wrong
	 */
	protected void convertInstance(Instance instance) throws Exception {
		double[] newVals = new double[getOutputFormat().numAttributes()];

		for (int i = 0; i < m_SelectedAttributes.length; i++) {
			int current = m_SelectedAttributes[i];
			newVals[i] = instance.value(current);
		}	    

		push (new DenseInstance(instance.weight(), newVals));

	}

	
	public boolean batchFinished() throws Exception {

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		//if (!isFirstBatchDone()) {
		if (!isOutputFormatDefined()) {

			Instances m_trainInstances = getInputFormat();
			m_trainInstances.deleteWithMissingClass();
			
			int m_numAttribs = m_trainInstances.numAttributes();

			System.out.println("num of atts:" + m_numAttribs);
			int m_numObjects = m_trainInstances.numInstances();
			System.out.println("num of objects:" + m_numObjects);
			
			
			BitSet reduct = new BitSet(m_numAttribs);

			BitSet objs = new BitSet(m_numObjects);
			
			AttributeStats cs = m_trainInstances.attributeStats(m_trainInstances.classIndex());
			
			int [] classDistribs = cs.nominalCounts;
			
			float val;

			//get class distribution for the data
			for(int j=0; j < classDistribs.length; j++){
			//System.out.println("classDistribs[" + j +"] = " + classDistribs[j]);
			val = (float)classDistribs[j]/m_numObjects;
			//System.out.println("val = " + val);
			//disregard any classes that represent less than 7% of the dataset
			if (val < 0.07) classDistribs[j] = 0;
			//System.out.println("New Class " + j + " "  + classDistribs[j]);	
			}
			

			m_Similarity.setInstances(m_trainInstances);
			m_DecisionSimilarity.setInstances(m_trainInstances);
			m_SimilarityEq.setInstances(m_trainInstances);
			m_composition = m_Similarity.getTNorm();

			m_Measure.set(m_Similarity,m_DecisionSimilarity,m_TNorm,m_composition,m_Implicator,m_SNorm,m_numObjects-1,m_numAttribs,m_trainInstances.classIndex(),m_trainInstances);

			reducer = new JohnsonReducer2(m_trainInstances, m_SNorm, m_Implicator);
			
			
			
			//Get the full clause list
			clauseList = ((FuzzyDiscernibilityBiReduct)m_Measure).calculateMatrix();
			System.out.println("Size of clause list: " + clauseList.size());


			while (clauseList.size() > 0)
			{
				//Pick a feature first
				int featFreq = reducer.heuristicPick(clauseList);
				//System.out.println("line 319");
				reduct.set(featFreq); //sets that feature in the reduct BitSet
				reducer.setReduct(reduct);
				clauseList = reducer.updateTrue(clauseList); //updates the current clause list
			
				//System.out.println("FeatFreq: " + featFreq);
				System.out.println("ClauseList Size (after feature selection):  " + clauseList.size());
				if(clauseList.size()==0) break; 
				
				//object frequency driven search for Bireducts
				if (m_objectFreq){
				//...then pick an object
				int[] objectFreq = reducer.countOccurence(clauseList);
				int highest = reducer.findHighestFrequency(objectFreq);
				objs.set(highest);	
				clauseList = reducer.updateTrueByObject(clauseList, highest);
				
				System.out.println("ClauseList Size (after object selection): " + clauseList.size());
				//System.out.println("Highest obj occ: " + highest);
				}
				
				//else do it based on the class distribution via roulette wheel selection 
				else{
					    //Class of object to select
						int classOfObject = -1;
						classOfObject = reducer.rouletteWheel(classDistribs);
						//System.out.println("roulette selected: " + classOfObject);
						//Get an object index for that class and the associated clause
						int [] indxAndClause = reducer.getIndxAndClause(clauseList, classOfObject);						
						//remove the clause of the class selected by indxAndClause
						clauseList = reducer.removeClauseByIndx(clauseList, indxAndClause[1]);
						System.out.println("ClauseList Size (after object selection):  " + clauseList.size());
						//set the object in the object list
						objs.set(indxAndClause[0]);
									
					
					
					
				} //end else

			}

			//System.out.println("class index: "+ m_trainInstances.classIndex());
			
			//Set the class index to true for any returned reduct
			reduct.set(m_trainInstances.classIndex());

			m_SelectedAttributes = new int [reduct.cardinality()];
			m_ObjectsToRemove = new int[objs.cardinality()];

			System.out.println("Reduct: " + reduct.toString());
			System.out.println("Object set: " + objs.toString());
			
			BitSet covering = (BitSet) objs.clone();
			covering.flip(0, m_numObjects);
			
			System.out.println("Covering set: " + covering);	
			System.out.println("BiReduct: " + reduct + ", " + covering);
			

			// store selected feats in array			
			int count = 0;
			//iterate over BitSet and get 'set' bits and add to array
			for (int i=0; i<m_numAttribs; i++)
			{

				if (reduct.get(i)) 
				{
					//System.out.println("This bit is true: " + i);
					m_SelectedAttributes[count] = i;
					count++;		

				}
			}

			//System.err.println("Count: "+count);
			
			// store selected objs in array
			int count1 = 0;
			//iterate over BitSet and get 'set' bits and add to array
			for (int i=0; i<m_numObjects; i++)
			{

				if (objs.get(i)) 
				{
					//System.out.println("This bit is true: " + i);
					m_ObjectsToRemove[count1] = i;
					count1++;		

				}
			}

			//Set the output format
			setOutputFormat();

//			Instances copyFS = new Instances(getOutputFormat(),0);
//			Instances copyObs = new Instances(getOutputFormat(),0);

		//	Instances copyFS = new Instances(getInputFormat(),0);


			// Convert pending input instances
			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				if (!objs.get(i))
				convertInstance(getInputFormat().instance(i));
			}

			//flushInput();
			
			// Set of instances for copying



			/*for (int l=0; l < m_ObjectsToRemove.length; l++){

				int currentOb = m_ObjectsToRemove [l];
				for (int m=0; m< getInputFormat().numInstances(); m++ )
					if (currentOb != m){
						copyObs.add(copyFS.get(l));
					}
			}


			for(int k=0; k < copyFS.size(); k++){

				push(copyFS.get(k));
			}*/













			/*Instances copy = new Instances(copy, m_numObjects);

			ArrayList <Integer> featsToSelect = new ArrayList();

			for (int i=0; i<m_numAttribs-1; i++)
			{
			 if (reduct.get(i)) 
				 {
				 featsToSelect.add(i);

				 System.out.println("This bit is true: " + i);

				 }
			}
			System.out.println("featsToSelect size: " + featsToSelect.size());
			java.util.Iterator j = featsToSelect.iterator();

			while(j.hasNext()) {
				System.out.println();
				copy.insertAttributeAt(m_trainInstances.get((Integer)j.next))),  );
				System.out.println("Here2");
			}*/

			//Then remove the objects


			/*ArrayList <Integer> objsToSelect = new ArrayList();

			for (int i=0; i<objs.size(); i++)
			{
			 if (objs.get(i)) objsToSelect.add(i); 
			}

			java.util.Iterator h = objsToSelect.iterator();		
			while(h.hasNext()) {

				getInputFormat().remove((Integer) h.next());
			}*/

			//			for (int k = 0; k < getInputFormat().numInstances(); k++) {
			//				push(getInputFormat().instance(k));
			//				System.out.println("Here3");
			//}





		}
		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
	}


}

