
<div align="center">
<h1>6 DOF UR3 robot Manipulation</h1>
A simple interactive UR3 robot arm simulation - This project is made to learn the basic integration of DH parameters, IK, and FK into a simple 3D model arm using just Python and matlplot animation function
<h4> <i> EECE 5250 Robot Mechanics and control, Northeastern University, Boston </i></h4>

[[Report](https://drive.google.com/file/d/1VCmS2vxOOsYXLE1RkLgm6P8pIUAeT4Tb/preview)]

<p float="center">
  <img src="Project 2/animation.gif" width="60%" />
</p>
</div>


# Abstract
This project develops a complete kinematic modeling and task-space motion framework for the 6-DOF UR3 collaborative robot manipulator. Using published Denavit–Hartenberg (DH) parameters, an analytical forward kinematics model is derived and implemented to compute end-effector pose (position and orientation) from joint angles. To solve inverse kinematics numerically, a geometric Jacobian formulation is implemented and used within a damped Newton–Raphson (Newton-based) iterative solver, where pose error is defined in both translation and rotation (with orientation error computed via an SO(3) logarithm-map representation). Task-space trajectories are specified directly as Cartesian waypoint sequences at approximately 1 mm resolution while maintaining a fixed end-effector orientation, including planar infinity, hexagonal, star-shaped, and 3D elliptical paths. Workspace reachability is also analyzed via Monte Carlo sampling over random joint configurations to generate a point-cloud representation of the robot’s reachable set.


# Requirements & Getting started
Install the requirements using `pip`. You might want to install these specific versions; later versions of matplotlib might have trouble loading the 3D simulation:
```
pip install -r requirements.txt
```

This simulation doesnt require heavy CPU / GPU its a simple python code to make a 3D arm simulation using the matplotlib animation function, make sure to check out this link to learn about the matplotlib animation function - [Matlplotlib Animation](https://matplotlib.org/stable/api/animation_api.html) & these are few opensource sample codes for working on your first matplotlib animation - [Sample codes](https://github.com/topics/matplotlib-animation).

The robot link lengths are taken from the online Universal Robotics DH parameters for calculating the dynamics and kinmatics  - [here](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/), You can see the full calculation of the FK and IK in the paper attached above.
