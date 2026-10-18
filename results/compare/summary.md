| Metric | Mink (off-the-shelf) | DLS-IK (closed-loop) | QP-IK (joint-limit constraints) |
| --- | --- | --- | --- |
| Success rate | 92% | 92% | 80% |
| Mean planning time (s) | 1.041 | 0.300 | 0.262 |
| Planning IK iterations / episode (mean) | 1144 | 1254 | 1621 |
| IK iterations / converged waypoint (mean) | 2.6 | 3.6 | 4.1 |
| IK compute per control step (ms) | 3.33 | 0.25 | 0.20 |
| Min robot-table clearance (mm) | 13.9 | 11.0 | 0.9 |
| Min robot-tray clearance (mm) | 4.8 | 1.4 | 6.1 |
| Min cube-tray clearance (mm) | 21.4 | 21.5 | 21.6 |
| Min joint-limit margin (rad) | 0.486 | 0.480 | 0.481 |
| Joint-limit hits (steps) | 0 | 0 | 0 |
| Velocity-limit violations (steps) | 0 | 0 | 0 |
| Peak joint speed / limit | 0.72 | 0.43 | 0.43 |
| Max TCP tracking lag (mm) | 24.7 | 58.4 | 157.0 |
| Max settled TCP error (mm) | 3.4 | 4.8 | 5.0 |
| Extra settle time / episode (s, mean) | 0.00 | 0.02 | 0.05 |
| Failures | planned_pose_collision: 1, ik_no_solution: 1 | planned_pose_collision: 1, ik_no_solution: 1 | ik_no_solution: 2, planned_pose_collision: 2, collision: 1 |
