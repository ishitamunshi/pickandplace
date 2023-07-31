# Type help("robolink") or help("robodk") for more information
# Press F5 to run the script
# Documentation: https://robodk.com/doc/en/RoboDK-API.html
# Reference:     https://robodk.com/doc/en/PythonAPI/index.html
# Note: It is not required to keep a copy of this file, your python script is saved with the station
from robolink import *    # RoboDK API
from robodk import *      # Robot toolbox
RDK = Robolink()

#------ CONSTANT ------
MECHANISM_NAME = 'Conveyor In'
PART_TRAVEL_MM = -150

mechanism = RDK.Item(MECHANISM_NAME,itemtype=ITEM_TYPE_ROBOT)

if mechanism.Valid():
    mechanism.MoveJ(mechanism.Joints() + PART_TRAVEL_MM)
