# You can also use the new version of the API:
from robodk import robolink  # RoboDK API
from robodk import robomath  # Robot toolbox

RDK = robolink.Robolink()
flag = 0
# Forward and backwards compatible use of the RoboDK API:
# Remove these 2 lines to follow python programming guidelines
from robodk import *  # RoboDK API
from robolink import *  # Robot toolbox

# Link to RoboDK
# RDK = Robolink()
robot = RDK.Item('UR10e')

# Notify user:

angle1 = [-114.477860, -89.060865, -111.974016, 21.034881, 114.637904, 0.000000]  # black
angle2 = [-50.219353, -65.861277, -129.524840, 15.386117, 50.379396, -0.000000]  # green
angle3 = [-13.083087, -97.528329, -102.949059, 20.477388, 13.243123, -0.000000]  # blue

while (flag == 0):
    with open("filename.txt", "r") as file:

        file_contents = file.read()
    print(file_contents)
    if file_contents == 'gear':
        robot.MoveJ(angle1)
        flag = 1
    elif file_contents == 'piston':
        robot.MoveJ(angle2)
        flag = 1
    elif file_contents == 'rod':
        robot.MoveJ(angle3)
        flag = 1
