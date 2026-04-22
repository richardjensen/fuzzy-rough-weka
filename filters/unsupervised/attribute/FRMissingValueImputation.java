/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.filters.unsupervised.attribute;

import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

import weka.classifiers.fuzzy.*;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.UnsupervisedFilter;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;

/**
 * <!-- globalinfo-start --> Replaces all missing values for nominal and numeric
 * attributes in a dataset 
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -unset-class-temporarily
 *  Unsets the class index temporarily before the filter is
 *  applied to the data.
 *  (default: no)
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author rkj
 * @version $Revision$
 */
public class FRMissingValueImputation extends PotentialClassIgnorer implements
UnsupervisedFilter, OptionHandler {

	/** for serialization */
	static final long serialVersionUID = 3495683109913409867L;

	/** The modes and means */
	private double[] m_ModesAndMeans = null;

	protected FuzzyRoughClassifier classifier = new FuzzyRoughNN();

	/**
	 * Returns a string describing this filter
	 *
	 * @return a description of the filter suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Replaces all missing values for nominal and numeric attributes in a "
				+ "dataset. The class attribute is skipped by default.";
	}

	/**
	 * Returns the Capabilities of this filter.
	 *
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enableAllAttributes();
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableAllClasses();
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);

		return result;
	}


	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(5);

		newVector.addElement(new Option(
				"\t.",
				"C", 1, "-C <num>"));


		return newVector.elements();
	}
	
	/**
	 * Parses and sets a given list of options. <p/>
	 *
	   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * 
	   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 *
	 **/
	public void setOptions (String[] options)
			throws Exception {

		String optionString = Utils.getOption('C', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid FuzzyRoughClassifier specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setClassifier( (FuzzyRoughClassifier) Utils.forName( FuzzyRoughClassifier.class, className, moreOptions) );
		}
		else {
			setClassifier(new FuzzyRoughNN());
		}

	}
	
	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-C");
		result.add((classifier.getClass().getName() + " " +
				Utils.joinOptions(classifier.getOptions())).trim());


		return result.toArray(new String[result.size()]);
	}

	public void setClassifier(FuzzyRoughClassifier c) {
		classifier = c;
	}
	
	public FuzzyRoughClassifier getClassifier() {
		return classifier;
	}
	
	/**
	 * Sets the format of the input instances.
	 *
	 * @param instanceInfo an Instances object containing the input instance
	 *          structure (any instances contained in the object are ignored -
	 *          only the structure is required).
	 * @return true if the outputFormat may be collected immediately
	 * @throws Exception if the input format can't be set successfully
	 */
	public boolean setInputFormat(Instances instanceInfo) throws Exception {

		super.setInputFormat(instanceInfo);
		setOutputFormat(instanceInfo);
		return true;
	}

	/**
	 * Input an instance for filtering. Filter requires all training instances be
	 * read before producing output.
	 *
	 * @param instance the input instance
	 * @return true if the filtered instance may now be collected with output().
	 * @throws IllegalStateException if no input format has been set.
	 */
	public boolean input(Instance instance) {

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		if (m_NewBatch) {
			resetQueue();
			m_NewBatch = false;
		}
		if (m_ModesAndMeans == null) {
			bufferInput(instance);
			return false;
		} else {
			convertInstance(instance);
			return true;
		}
	}

	/**
	 * Signify that this batch of input to the filter is finished. If the filter
	 * requires all instances prior to filtering, output() may now be called to
	 * retrieve the filtered instances.
	 *
	 * @return true if there are instances pending output
	 * @throws IllegalStateException if no input structure has been defined
	 */
	public boolean batchFinished() {

		Instances m_trainInstances = getInputFormat();

		if (m_trainInstances == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		try {
			classifier.buildClassifier(m_trainInstances);

			//for (int i=0; i<m_trainInstances.numInstances(); i++) {
			Iterator<Instance> it = m_trainInstances.iterator();

			while (it.hasNext()) {
				convertInstance(it.next());
			}
		}
		catch (Exception e) {System.err.println(e);}


		// Free memory
		flushInput();

		m_NewBatch = true;
		return (numPendingOutput() != 0);
	}

	/**
	 * Convert a single instance over. The converted instance is added to the end
	 * of the output queue.
	 *
	 * @param instance the instance to convert
	 */
	private void convertInstance(Instance instance) {
		boolean hasMissing = instance.hasMissingValue();

		if (hasMissing) {
			// find the missing value(s)
			for (int j = 0; j < instance.numAttributes(); j++) {
				if (instance.isMissing(j)) {
					classifier.setAltClassIndex(j); // set this to be predicted

					// debug
					//System.out.println("Original instance: "+instance.toString());
					
					try {
						double[] dist = classifier.distributionForInstance(instance);
						
						if (instance.attribute(j).isNominal()) {
							double max = 0;
							int maxIndex = 0;

							// find the best label
							for (int i = 0; i < dist.length; i++) {
								if (dist[i] > max) {
									maxIndex = i;
									max = dist[i];
								}
							}
							
							// set the value of the instance to the determined label
							instance.setValue(j, instance.attribute(j).value(maxIndex));

						}
						else { // numeric attribute
							instance.setValue(j,dist[0]);
						}
						
						// debug
						//System.out.println("Changed instance:  "+instance.toString()+" -> "+dist[0]);
						//System.out.println();

					}
					catch (Exception e) {System.err.println(e);}
				}

			}
		}

		//inst.setDataset(instance.dataset());
		push(instance, !hasMissing);
	}



	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain arguments to the filter: use -h for help
	 */
	public static void main(String[] argv) {
		runFilter(new FRMissingValueImputation(), argv);
	}
}
