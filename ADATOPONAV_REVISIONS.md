# AdaTopoNav implementation and paper alignment

This note defines what the current code supports and which claims are safe to
make in the paper. Numerical performance claims still require completed,
logged trials.

## Implemented behavior

- `online_mapper.py` approximately synchronizes RGB and odometry messages.
- A revisit inside `snap_threshold` updates the active node without adding a
  node.
- Leaving a historical node uses `dense_distance`; extending the newest node
  uses `sparse_distance`. This state test does not detect corridor straightness.
- A branch parent is the physically nearest eligible historical node inside
  `parent_radius`.
- Each new node has exactly one parent edge. The result is a branch-preserving
  tree, not a mesh and not a loop-closing graph.
- Global routing computes a shortest path within the constructed tree.
- DINOv2 cosine similarity is an appearance-continuity heuristic. It is not a
  calibrated geometric visibility or occlusion detector.
- Goal completion requires the estimated goal node, metric proximity to its
  stored pose, and repeated confirmation. Process termination is coordinated
  with a ROS status topic rather than `pkill`.
- The low-level module is a clipped waypoint-to-velocity controller, not a PD
  controller.

## Required paper wording changes

Use “empirically selected in the evaluated layouts” for the 0.65 m, 1.20 m,
1.50 m, 0.85, 0.44 rad, 0.6, and 1.5 parameters unless a sensitivity study is
reported.

Replace:

- “approximately straight corridor” with “continued extension from the newest
  node”;
- “detects geometric occlusion” with “detects low appearance continuity”;
- “globally optimal route” with “shortest route in the constructed tree”;
- “PD controller” with “clipped waypoint-to-velocity controller”;
- “collision-avoidance command” with “waypoint-derived velocity command”.

Do not report the disabled `\\iffalse` results. A one-parent tree with more
than one node has average degree `2(|V|-1)/|V|`, which is always below 2; an
average degree of 2.7 cannot be produced by this mapper.

## Experimental reporting

`deployment/src/evaluate_navigation_trial.py` records the trial ID,
environment, method, seed, success, debounced collisions, elapsed time,
travelled distance, optional shortest-path distance, final goal distance, and
termination reason. A collision topic and its ROS message type must be
configured for the simulator in use. Software failures must remain in the
trial table rather than being silently excluded.

Do not fill the paper tables until all planned paired trials have completed.
The existing `deployment/src/test_results` logs include timeouts and failed
runs and are not evidence for the placeholder results in the disabled paper
block.

## Reproducibility checklist

- Commit the Gazebo worlds, robot/contact-sensor configuration, fixed
  start-goal list, and reset procedure used for the paper.
- Record the exact NoMaD checkpoint checksum and DINOv2 version.
- Report hardware, mean/percentile inference latency, and feature-extraction
  latency before claiming negligible overhead.
- Publish raw per-trial CSV files and the script that produces aggregate SR,
  collision, time, and SPL values.
