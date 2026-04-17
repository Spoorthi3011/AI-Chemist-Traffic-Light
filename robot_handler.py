from examples.utils.UR_Functions import URfunctions as URControl
from utils.coordinate_utility import coordinate_util as coord_util

HOME = [1.3651890754699707, -1.5046540063670655, 1.8308394590960901, -1.913860937158102, -1.567221466694967, 2.9111227989196777]

SLOW_ACC = 0.25
SLOW_SPEED = 0.8
ROT = 0.02
FAST_ACC = 1
FAST_SPEED = 2.5


class robot(URControl):
    def __init__(self, ip: str, port: int):
        self.robot = self._init_robot()
        self.ip_addr = ip
        self.port = port
        self.current_position = HOME

    def _init_robot(self):
        self.robot = URControl(ip="192.168.0.2", port=30003)
        print(self.robot.get_current_joint_positions().tolist())
        return self.robot

    def return_home(self):
        self.robot.go_home()
    
    def get_joints(self):
        joint_pos = self.robot.get_joint_positions()
        ## get just the joint positions from here
        return joint_pos
    
    def move_joints_fast(self, position: list):
        self.robot.move_joint_list(position,  FAST_SPEED, FAST_ACC,ROT)

    def move_joints_slow(self, position: list):
        self.robot.move_joint_list(position,  SLOW_SPEED, SLOW_ACC, ROT)
    
   