package weka.attributeSelection.ann;

import weka.core.Instances;

import java.util.*;

public class NavigableSmallWorld implements AnnInterface {
    private final int M;
    private final AnnUtils.DistanceFunction distanceFn;
    private List<List<Integer>> graph;
    private double[][] data;
    private int entryPoint = 0;

    public NavigableSmallWorld(int M, AnnUtils.DistanceFunction distanceFn) {
        this.M = M;
        this.distanceFn = distanceFn;
    }

    private List<AnnUtils.QueueItem> searchLayer(int[] entryPoints, double[] queryVec, int k) {
        if (data == null || data.length == 0) return Collections.emptyList();

        Set<Integer> visited = new HashSet<>();
        for (int ep : entryPoints) visited.add(ep);

        // Min-priority queue for candidates to visit (closest first)
        PriorityQueue<AnnUtils.QueueItem> candidates = new PriorityQueue<>(Comparator.comparingDouble(a -> a.priority));
        // Max-priority queue for results (farthest first, so we can discard)
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

            for (int neighborIdx : graph.get(current.index)) {
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

        this.graph = new ArrayList<>(data.length);
        for (int i = 0; i < data.length; i++) {
            graph.add(new ArrayList<>());
        }

        // Sequentially insert points
        for (int i = 1; i < data.length; i++) {
            List<AnnUtils.QueueItem> neighbors = searchLayer(new int[]{entryPoint}, data[i], M);
            for (AnnUtils.QueueItem neighbor : neighbors) {
                int neighborIdx = neighbor.index;
                graph.get(i).add(neighborIdx);
                graph.get(neighborIdx).add(i);
            }
        }
    }

    @Override
    public int[] query(double[] vector, int k) {
        if (data == null || data.length == 0) return new int[0];
        
        // Use k if provided, otherwise a default search size
        int searchSize = (k > 0) ? k : 50; 
        
        List<AnnUtils.QueueItem> resultItems = searchLayer(new int[]{entryPoint}, vector, searchSize);
        return resultItems.stream().mapToInt(item -> item.index).toArray();
    }
}