# pickandplace

The main objective was to create a robot simulation using RoboDK software for an industrial pick-and-place application.

In the simulation, a conveyor was used to transport various objects of different classes, positioned randomly. Instead of manually specifying the coordinates for the robot's pick action, I employed computer vision techniques using OpenCV. Specifically, I utilized YOLOv8 to train a model that could detect these objects and determine their precise coordinates with respect to a predefined reference frame. These were then used to move the gripper to the correct location and pick the object as well as segregate it while placing it in different bins according to their classes.


https://github.com/ishitamunshi/pickandplace/assets/110255438/5a149a0b-9390-4e90-a6b5-e3cd217ba786

