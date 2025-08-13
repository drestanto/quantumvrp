# 8\_google\_or-tools/

This directory contains the `or-tools_vrp.py` script, which demonstrates how to solve a Vehicle Routing Problem (VRP) using Google's open-source OR-Tools library. This script leverages the pre-processed data from the previous steps, specifically the complete graphs with modified distances, to find an optimal route for a given cluster of locations.

-----

## Contents

  * `or-tools_vrp.py`:
      * **Purpose**: This is the main script for solving the VRP using Google OR-Tools. It reads a pre-computed complete graph (from `cluster_complete_graphs.json`) for a specific cluster. It then sets up the OR-Tools model, including the distance matrix, and uses the `pywrapcp` and `routing_enums_pb2` modules to define a solver. The script employs a `FirstSolutionStrategy.PATH_CHEAPEST_ARC` to find an initial solution and then refines it using a `LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH` to find a near-optimal solution. Finally, it prints the optimal route and its total distance.
      * **Configuration**: Before running, you **must** edit the `cluster_type` and `target_id` variables within the script to specify which VRP cluster from `cluster_complete_graphs.json` you want to solve.
      * **Dependencies**: This script relies on `ortools`, `json`, `math`, and `time`.
      * **Usage**:
    <!-- end list -->
    ```bash
    python or-tools_vrp.py
    ```

-----

## Workflow

1.  **Preparation**: Ensure the `cluster_complete_graphs.json` file (from `7_incorporate_constraints/`) is available and contains the data for the cluster you wish to solve.
2.  **Configure Solver**: Open `or-tools_vrp.py` and set the `cluster_type` and `target_id` for the VRP instance you wish to solve.
3.  **Run Solver**: Execute `python or-tools_vrp.py`. The script will output the details of the solution, including the route and its total cost.

-----

## Notes

  * **OR-Tools**: Google OR-Tools is an open-source software suite for optimization problems, including VRP. It provides a wide range of algorithms for finding solutions.
  * **First Solution Strategy**: The `PATH_CHEAPEST_ARC` strategy is a heuristic used by OR-Tools to quickly generate an initial feasible solution. It builds a route by repeatedly adding the cheapest possible arc from the current node until a complete route is formed.
  * **Local Search Metaheuristic**: The `GUIDED_LOCAL_SEARCH` is a powerful metaheuristic that improves upon an initial solution. It works by penalizing certain solution features in order to guide the search out of local minima and explore other parts of the solution space. This helps in finding a better, more optimized final route.
  * **Performance**: The performance of the solver can be influenced by the choice of the first solution strategy and local search metaheuristic, and a time limit can be set to control the search duration.

-----

  * **Citations:**
      * **[1]** "OR-Tools: Software Google OR-Tools is a free and open-source software suite developed by Google for solving linear programming, mixed integer programming, constraint programming, vehicle routing, and related optimization problems. OR-Tools is a set of components written in C++ but provides wrappers for Java, .NET and Python." - from a Wikipedia article on OR-Tools.
      * **[2]** The `pywrapcp` module contains the core components for the OR-Tools constraint programming solver, including classes for variables, constraints, and the solver itself.
      * **[3]** The `routing_enums_pb2` namespace contains enumerations for various routing-related constants, such as those used for first solution strategies and local search metaheuristics.
      * **[4]** The `FirstSolutionStrategy.PATH_CHEAPEST_ARC` heuristic builds a route by repeatedly connecting the current node to the one that creates the cheapest possible route segment.
      * **[5]** The `LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH` is a metaheuristic that enhances local search algorithms by building up penalties to help the search escape from local minima and plateaus. It modifies the objective function to guide the search towards more promising regions.