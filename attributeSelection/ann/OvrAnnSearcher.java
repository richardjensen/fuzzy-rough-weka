package weka.attributeSelection.ann;

import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Manages multiple ANN indexes using a One-vs-Rest strategy.
 * One index is created for each class label in the dataset.
 */
public class OvrAnnSearcher {

    private final Supplier<AnnInterface> annFactory;
    private final Map<Double, AnnInterface> classIndexes = new HashMap<>();
    private final Map<Double, List<Integer>> classDataGlobalIndices = new HashMap<>();
    private Instances schema;

    public OvrAnnSearcher(Supplier<AnnInterface> annFactory) {
        this.annFactory = annFactory;
    }

    public void buildIndex(Instances data) {
        this.schema = new Instances(data, 0);
        int classIndex = data.classIndex();

        // 1. Partition data by class
        Map<Double, Instances> partitionedData = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double classValue = inst.classValue();
            
            partitionedData.computeIfAbsent(classValue, k -> new Instances(schema, 0)).add(inst);
            classDataGlobalIndices.computeIfAbsent(classValue, k -> new ArrayList<>()).add(i);
        }

        // 2. Build an ANN index for each partition
        for (Map.Entry<Double, Instances> entry : partitionedData.entrySet()) {
            double classValue = entry.getKey();
            Instances classInstances = entry.getValue();

            AnnInterface annIndex = annFactory.get();
            annIndex.buildIndex(classInstances);
            classIndexes.put(classValue, annIndex);
        }
    }

    /**
     * Finds nearest "enemies" by querying all indexes EXCEPT the one for the query instance's own class.
     * @param queryInstance The instance to find enemies for.
     * @return An array of global indices of candidate enemies.
     */
    public int[] queryEnemies(Instance queryInstance) {
        double queryClassValue = queryInstance.classValue();
        double[] queryVector = AnnUtils.instanceToVector(queryInstance);
        
        List<Integer> finalCandidates = new ArrayList<>();

        for (Map.Entry<Double, AnnInterface> entry : classIndexes.entrySet()) {
            double indexClassValue = entry.getKey();

            // Skip the index of the same class
            if (indexClassValue == queryClassValue) {
                continue;
            }

            AnnInterface annIndex = entry.getValue();
            List<Integer> globalIndicesMap = classDataGlobalIndices.get(indexClassValue);

            int[] localResults = annIndex.query(queryVector, -1); // -1 means return all candidates

            for (int localIndex : localResults) {
                finalCandidates.add(globalIndicesMap.get(localIndex));
            }
        }
        
        return finalCandidates.stream().mapToInt(i -> i).toArray();
    }
}