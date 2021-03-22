# Team Optimal Robot Controller

This repository contains code for the robot controller, it should be added as a
submodule to the [main Webots project
repository](https://github.com/IDP-L211/simulation). Documentation can be found
at https://idp-l211.github.io/controllers/.

The robot chooses the closest accessible target to drive to, or perform a scan
if no such target is available. When the robot reaches a target, it will check
the colour using two LDRs. If the colour is wrong, the robot will send the
confirmed position to the other robot.

<p align="center">
  <img alt="Overall Flowchart" src="docs/figs/overall_flow.svg" width="300">
</p>

Custom classes for devices are created by inheriting from classes in Webots C
interface, which are then composed together in a robot class. Various
subroutines are also handle by child classes (e.g. `TargetingHandler`). The
class diagram is shown below
![Class Diagram](docs/figs/uml.svg)

## Algorithms and Subroutines

DBSCAN clustering algorithm is used in the scanning subroutine to calculate
block positions from filtered data from IR sensor and GPS. This means the
effect of noise is minimised, and the target positions are relatively accurate.

The targeting algorithm picks the closest block of a valid class that has a
clear path to it. A clear path is determined by creating a rectangle from the
robot to the target and checking no other object is inside it. It will start
the check from the closest potential target and continue until a valid target
is found. When driving to the selected target, an active collision avoidance
algorithm (with a similar working principle) diverts the robot to a different
target if necessary.

A passive collision avoidance algorithm is used when the robot is not moving
towards a block. It calculates the distance to each known object and then does
a weighted sum of the current target angle and the angles to avoid each object
(i.e. right angle away from it). A recursive component is added to combine
small obstructions into larger ones to avoid clusters of blocks.

When the robot is far from the target, proportional combined with open loop
control is used to achieve a velocity. When nearby, a proportional distance
controller is used. A non-linear controller which uses both error and
derivative are used to control the angle. The two controllers produced forward
and rotational velocities respectively, which were combined to give the overall
motor velocities.

<p align="center">
  <img alt="Motion" src="docs/figs/overall_flow.svg" width="250">
  <img alt="Scanning" src="docs/figs/scan_flow.svg" width="200">
  <img alt="Targeting" src="docs/figs/target_flow.svg" width="200">
</p>

## Development

It is useful to set up a proper development environment with the provided
`controller` module (a C interface) included in your linter/IDE and set the
robot controller to `<extern>` in Webots for debugging. More information at
https://cyberbotics.com/doc/guide/using-your-ide?tab-language=python. The main
steps for PyCharm are:

1. Configure the virtual environment in _Project Interpreter_
2. Add the correct Webots `controller` directory (depending on your system and
   Python configuration, `controller/_controller.so` is slightly different) as
   _Content Root_ under _Project Structure_
3. Create a _Run/Debug Configuration_, setting the [relevant environment
   variables](https://cyberbotics.com/doc/guide/running-extern-robot-controllers?tab-language=python)

If you prefer to run controller directly from Webots, have a look at
https://cyberbotics.com/doc/guide/using-python. This is especially important
for people using macOS and Homebrew, you may need to set the full path to your
Python interpreter (get it by `which python3`).
