from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from synthnova_config import PhysicsSimulatorConfig, RobotConfig
import numpy as np
import time
from pathlib import Path

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps)

class BotTest1():



    def _setup_interface(self):
        galbot_interface_config = GalbotInterfaceConfig()

        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = "galbot_one_charlie"
        # Enable modules
        galbot_interface_config.modules_manager.enabled_modules.append("right_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("leg")
        galbot_interface_config.modules_manager.enabled_modules.append("head")
        galbot_interface_config.modules_manager.enabled_modules.append("chassis")

        galbot_interface_config.right_arm.joint_names = [
            f"{robot_name}/right_arm_joint1",
            f"{robot_name}/right_arm_joint2",
            f"{robot_name}/right_arm_joint3",
            f"{robot_name}/right_arm_joint4",
            f"{robot_name}/right_arm_joint5",
            f"{robot_name}/right_arm_joint6",
            f"{robot_name}/right_arm_joint7",
        ]

        galbot_interface_config.left_arm.joint_names = [
            f"{robot_name}/left_arm_joint1",
            f"{robot_name}/left_arm_joint2",
            f"{robot_name}/left_arm_joint3",
            f"{robot_name}/left_arm_joint4",
            f"{robot_name}/left_arm_joint5",
            f"{robot_name}/left_arm_joint6",
            f"{robot_name}/left_arm_joint7",
        ]

        galbot_interface_config.leg.joint_names = [
            f"{robot_name}/leg_joint1",
            f"{robot_name}/leg_joint2",
            f"{robot_name}/leg_joint3",
            f"{robot_name}/leg_joint4",
        ]

        galbot_interface_config.head.joint_names = [
            f"{robot_name}/head_joint1",
            f"{robot_name}/head_joint2"
        ]

        galbot_interface_config.chassis.joint_names = [
            f"{robot_name}/mobile_forward_joint",
            f"{robot_name}/mobile_side_joint",
            f"{robot_name}/mobile_yaw_joint",
        ]

        galbot_interface = GalbotInterface(
            galbot_interface_config=galbot_interface_config,
            simulator=self.simulator
        )
        galbot_interface.initialize()

        self.interface = galbot_interface

    def _init_pose(self):
        # Init head pose
        head = [0.0, 0.0]
        self._move_joints_to_target(self.interface.head, head)

        # Init leg pose
        leg = [0.43, 1.48, 1.07, 0.0]
        self._move_joints_to_target(self.interface.leg, leg)

        # Init left arm pose
        left_arm = [
            0.058147381991147995,
            1.4785659313201904,
            -0.0999724417924881,
            -2.097979784011841,
            1.3999720811843872,
            -0.009971064515411854,
            1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.left_arm, left_arm)

        # Init right arm pose
        right_arm = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.right_arm, right_arm)

    def setup_sim(self):
        #Create sim config
        my_config = PhysicsSimulatorConfig()

        # Instantiate the simulator
        self.simulator = PhysicsSimulator(my_config)

        # Add default ground plane if you need
        self.simulator.add_default_scene()

        # Add robot
        self.robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_charlie",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robot")
            .joinpath("galbot_one_charlie_description")
            .joinpath("galbot_one_charlie.xml")
            .as_posix(),
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )
        self.robot_path = self.simulator.add_robot(self.robot_config)

        # Initialize the simulator
        self.simulator.initialize()

    def setup_scence(self):
        print("scence")

    # def init_interface(self):

    #     # Initialize the galbot inte    rface
    #     galbot_interface_config = GalbotInterfaceConfig()
    #     # Enable the modules
    #     galbot_interface_config.modules_manager.enabled_modules.append("chassis")
    #     galbot_interface_config.chassis.joint_names = [
    #         f"{self.robot_config.name}/mobile_forward_joint",
    #         f"{self.robot_config.name}/mobile_side_joint",
    #         f"{self.robot_config.name}/mobile_yaw_joint",
    #     ]
    #     # Bind the simulation entity prim path to the interface config
    #     galbot_interface_config.robot.prim_path = self.robot_path
    #     self.galbot_interface = GalbotInterface(
    #         galbot_interface_config=galbot_interface_config,
    #         simulator=self.simulator
    #     )
    #     self.galbot_interface.initialize()

    def check_movement_complete(self, target, atol):
        current_positions = self.interface.chassis.get_joint_positions()
        return np.allclose(current_positions, target, atol=atol)
    
    def _move_joints_to_target(self, module, target_positions, steps=200):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def follow_path_callback(self):
        
        # if there is a movement command in queue
        if (len(self.fifoPath) != 0):
            
            # load command from queue
            target = self.fifoPath[0]
            # print(self.check_movement_complete(target, 0.1) if self.check_movement_complete(target, 0.1) else "") 
            if (self.check_movement_complete(target, 0.1)): # if target has been reached within 0.1 tolerance
                
                print("pop")
                
                self.fifoPath.pop(0) # remove element from queue
                print(self.fifoPath)
                # print(x)
                self.moving = False
                if (len(self.fifoPath) != 0): # if another element in queue
                    target = self.fifoPath[0]
                    self.moveGeneric(target) # move to target
                    self.moving = True
                    # self.follow_path_callback() # and then run the loop again

    def moveGeneric(self,target):
        current_joint_positions = self.interface.chassis.get_joint_positions()
        positions = interpolate_joint_positions(current_joint_positions,target,200)
        joint_trajectory = JointTrajectory(positions=positions)
        self.interface.chassis.follow_trajectory(joint_trajectory)

    def moveForwards(self,step):
        new_target = list(self.fifoPath[-1])
        new_target[0] += step
        self.fifoPath.append(new_target)


    
    def moveBackwards(self,step):
        self.moveForwards((step if step < 0 else (-step)))

    def moveLeft(self,step):
        new_target = list(self.fifoPath[-1])
        new_target[1] += step
        self.fifoPath.append(new_target)

    
    def moveRight(self,step):
        self.moveLeft((step if step < 0 else (-step)))

    def pitchYawLeft(self,step):
        new_target = list(self.fifoPath[-1])
        new_target[2] += step
        self.fifoPath.append(new_target)


    def pitchYawRight(self,step):
        self.pitchYawLeft((step if step < 0 else (-step)))
    


    def main(self):

        self.setup_sim()
        #self.init_interface()
        self._setup_interface()

        self._init_pose()

        self.simulator.add_physics_callback("follow_path_callback",self.follow_path_callback)
        self.moving = False
        self.fifoPath = [[0,0,0],[9,0,0],[3,0,0]]
        
        # self.fifoPath = []
        # self.moveBackwards(2)
        # self.moveForwards(2)
        
        # Start the simulation
        self.simulator.step()



        # # Get current joint positions
        # current_joint_positions = self.galbot_interface.chassis.get_joint_positions()

        # # Define target joint positions
        # target_joint_positions = [1, 6, 1.5]

        # # Interpolate joint positions
        # positions = interpolate_joint_positions(
        #     current_joint_positions, target_joint_positions, 5000
        # )
        # # Create a joint trajectory
        # joint_trajectory = JointTrajectory(positions=positions)

        # # Follow the trajectory
        # self.galbot_interface.chassis.follow_trajectory(joint_trajectory)

        # Run the display loop
        while (True):
            self.simulator.step()

        # Close the simulator
        self.simulator.close()

if __name__ == "__main__":
    test = BotTest1()
    test.main()