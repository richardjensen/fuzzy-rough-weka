package weka.attributeSelection.ann;

import weka.core.Instances;

import java.util.*;

public class HNSW implements AnnInterface {
    private final int M;
    private final int efConstruction;
    private final AnnUtils.DistanceFunction distanceFn;
    private final double mL;
    private final Random rand = new Random();
    
    private double[][] data;
    private List<List<List<Integer>>> graphs; // Layer -> NodeId -> Neighbors
    private Integer entryPoint = null;
    private int maxLayer = -1;

    public HNSW(int M, int efConstruction, AnnUtils.DistanceFunction distanceFn) {
        this.M = M;
        this.efConstruction = efConstruction;
        this.distanceFn = distanceFn;
        this.mL = 1.0 / Math.log(this.M);
    }

    private int getRandomLayer() {
        return (int) Math.floor(-Math.log(rand.nextDouble()) * mL);
    }

    private List<AnnUtils.QueueItem> searchLayer(double[] queryVec, int[] entryPoints, int k, int layerNum) {
        Set<Integer> visited = new HashSet<>();
        for (int ep : entryPoints) visited.add(ep);

        PriorityQueue<AnnUtils.QueueItem> candidates = new PriorityQueue<>(Comparator.comparingDouble(a -> a.priority));
        PriorityQueue<AnnUtils.QueueItem> results = new PriorityQueue<>((a, b) -> Double.compare(b.priority, a.priority));

        for (int ep : entryPoints) {
            double dist = distanceFn.compute(queryVec, data[ep]);
            candidates.add(new AnnUtils.QueueItem(ep, dist));
            results.add(new AnnUtils.QueueItem(ep, dist));
        }

        while (!candidates.isEmpty()) {
            AnnUtils.QueueItem current = candidates.poll();
            double worstResultDist = results.isEmpty() ? Double.POSITIVE_INFINITY : results.peek().priority;

            if (current.priority > worstResultDist && results.size() >= k) {
                break;
            }

            List<Integer> neighbors = graphs.get(layerNum).get(current.index);
            for (int neighborIdx : neighbors) {
                if (!visited.contains(neighborIdx)) {
                    visited.add(neighborIdx);
                    double dist = distanceFn.compute(queryVec, data[neighborIdx]);
                    if (results.size() < k || dist < results.peek().priority) {
                        candidates.add(new AnnUtils.QueueItem(neighborIdx, dist));
                        results.add(new AnnUtils.QueueItem(neighborIdx, dist));
                        if (results.size() > k) {
                            results.poll();
                        }
                    }
                }
            }
        }

        List<AnnUtils.QueueItem> finalResults = new ArrayList<>(results);
        finalResults.sort(Comparator.comparingDouble(a -> a.priority));
        return finalResults;
    }

    @Override
    public void buildIndex(Instances instances) {
        this.data = AnnUtils.instancesToMatrix(instances);
        if (data.length == 0) return;

        this.graphs = new ArrayList<>();
        this.entryPoint = 0;
        this.maxLayer = getRandomLayer();
        for (int i = 0; i <= maxLayer; i++) {
            List<List<Integer>> newLayer = new ArrayList<>(data.length);
            for (int j=0; j < data.length; j++) newLayer.add(new ArrayList<>());
            graphs.add(newLayer);
        }

        for (int i = 1; i < data.length; i++) {
            int pointLevel = getRandomLayer();
            int currentEntryPoint = entryPoint;

            for (int level = maxLayer; level > pointLevel; level--) {
                List<AnnUtils.QueueItem> searchResult = searchLayer(data[i], new int[]{currentEntryPoint}, 1, level);
                if (!searchResult.isEmpty()) {
                    currentEntryPoint = searchResult.get(0).index;
                }
            }

            for (int level = Math.min(pointLevel, maxLayer); level >= 0; level--) {
                List<AnnUtils.QueueItem> neighbors = searchLayer(data[i], new int[]{currentEntryPoint}, efConstruction, level);
                List<Integer> selectedNeighbors = new ArrayList<>();
                for (int j = 0; j < Math.min(M, neighbors.size()); j++) {
                    selectedNeighbors.add(neighbors.get(j).index);
                }

                for (int neighborIdx : selectedNeighbors) {
                    graphs.get(level).get(i).add(neighborIdx);
                    graphs.get(level).get(neighborIdx).add(i);
                }
                
                if(!neighbors.isEmpty()) {
                    currentEntryPoint = neighbors.get(0).index;
                }
            }

            if (pointLevel > maxLayer) {
                for (int j = maxLayer + 1; j <= pointLevel; j++) {
                     List<List<Integer>> newLayer = new ArrayList<>(data.length);
                     for (int k=0; k < data.length; k++) newLayer.add(new ArrayList<>());
                     graphs.add(newLayer);
                }
                maxLayer = pointLevel;
                entryPoint = i;
            }
        }
    }

    @Override
    public int[] query(double[] vector, int k) {
        if (entryPoint == null) return new int[0];
        
        int efSearch = (k > 0) ? Math.max(k, 50) : 50;
        int currentEntryPoint = entryPoint;

        for (int level = maxLayer; level > 0; level--) {
            List<AnnUtils.QueueItem> searchResult = searchLayer(vector, new int[]{currentEntryPoint}, 1, level);
            if (!searchResult.isEmpty()) {
                currentEntryPoint = searchResult.get(0).index;
            }
        }

        List<AnnUtils.QueueItem> resultItems = searchLayer(vector, new int[]{currentEntryPoint}, efSearch, 0);
        
        // If a specific k is requested, trim the results. Otherwise, return all found candidates from the search.
        if (k > 0 && resultItems.size() > k) {
            resultItems = resultItems.subList(0, k);
        }
        
        return resultItems.stream().mapToInt(item -> item.index).toArray();
    }
}