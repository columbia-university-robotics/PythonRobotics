# CURC Self-Driving Motion Planning

## Approach

 - Drive the car manual to get a map (occupancy grid) of the track.
 - Run A* (`a_star_curcuit.py`) to get an array of waypoints; save these waypoints to a file (`rx.npy`, `ry.npy`).
 - Run Frenet Node to operate the car (passes a Twist to the inverse kinematics node) using waypoints as a guide.

## Sim2Real

At this time, the `frenet_node.py` is a simulation environment rather than a ROS node. To convert this to a ROS node, we need to (1) use `rospy.Rate()` and `rospy.is_shutdown()` instead of the simulation loop, (2) use the callbacks for the map and pose, (3) publish to the inverse kinematics node/topic instead of assuming perfect path following.
