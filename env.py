from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig
import numpy as np
import math
from pathlib import Path


def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps)

def printEnv(string):
    envName = "Mujoco"
    print("[" + envName +"] " + str(string))

class environment():

    def setup_sim(self):
        # Create sim config
        my_config = PhysicsSimulatorConfig()
        my_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(timestep=0.1)
        )
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
            .joinpath("galbot_one_charlie.xml"),
            position=[0, 0, 0], # x, y, yaw
            orientation=[0, 0, 0, 1]
        )
        self.robot_path = self.simulator.add_robot(self.robot_config)

        # Initialize the simulator
        self.simulator.initialize()

    def steup_scence(self):
        print("scence")

    def init_interface(self):

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


    def _move_joints_to_target(self, module, target_positions, steps=200):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    # generic joint position to target position check
    def _is_joint_positions_reached(self, module, target_positions):
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=0.1)
    
    def _init_pose(self):
        # Init head pose
        self.head = [0.0, 0.0]
        self._move_joints_to_target(self.interface.head, self.head)

        # Init leg pose
        self.leg = [0.43, 1.48, 1.07, 0.0]
        self._move_joints_to_target(self.interface.leg, self.leg)

        # Init left arm pose
        self.left_arm = [
            0.058147381991147995,
            1.4785659313201904,
            -0.0999724417924881,
            -2.097979784011841,
            1.3999720811843872,
            -0.009971064515411854,
            1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.left_arm, self.left_arm)

        # Init right arm pose
        self.right_arm = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.right_arm, self.right_arm)

        self.simulator.add_physics_callback("is_init_done", self._init_pose_done)
    
    def _init_pose_done(self):
        headState = False
        legState = False
        left_armState = False
        right_armState = False
        # if head has reached target
        if (headState | self._is_joint_positions_reached(self.interface.head, self.head)):
            headState = True
        
        # if leg has reached target
        if (legState | self._is_joint_positions_reached(self.interface.leg, self.leg)):
            legState = True

        # if left arm has reached target
        if (left_armState | self._is_joint_positions_reached(self.interface.left_arm, self.left_arm)):
            left_armState = True
        
        # if right arm has reached target
        if (right_armState | self._is_joint_positions_reached(self.interface.right_arm, self.right_arm)):
            right_armState = True
        
        # if all targets have been reached
        if (headState and legState and left_armState and right_armState):
            self.stepOffset = self.simulator.get_step_count() # set step offset
            printEnv("init done")
            self.initDone = True
            self.simulator.remove_physics_callback("is_init_done")

    def main(self):
        
        self.setup_sim()
        self.init_interface()

        self._init_pose()
        
        # Start the simulation
        self.simulator.step()


       # # Get current joint positions
       # current_joint_positions = self.galbot_interface.chassis.get_joint_positions() # [x, y, yaw]

       # # Define target joint positions
       # target_joint_positions = [1, 6, 1.5] # x, y , yaw

       # # Interpolate joint positions
       # positions = interpolate_joint_positions(
       #     current_joint_positions, target_joint_positions, 5000
       # )
       # # Create a joint trajectory
       # joint_trajectory = JointTrajectory(positions=positions)

       # # Follow the trajectory
       # self.galbot_interface.chassis.follow_trajectory(joint_trajectory)

        # action fifo queue
        self.simulator.add_physics_callback("follow_path_callback", self.follow_path_callback)
        self.moving = False
        self.fifoPath = [[0,0,0]] # path offset [x,y,yaw], yaw in radians

        self.initDone = False
        printEnv("sim is loading...")
        while(not self.initDone):
            # print("running step")
            self.simulator.step()
            # self.follow_path_callback()

        printEnv("sim started")
        
        self.step(0)

        self.step(2)

        print("done")
        # Run the display loop
        while (True):
            self.simulator.step()

        # Close the simulator
        synthnova_physics_simulator.close()
    
    def computeRobotPositionRelative(self):
        # the chassis coordinates are relative to where the robot starts
        # compute real coordinates from chassis offset
        robotLocation = self.interface.chassis.get_joint_positions()
        robotLocation = [0+robotLocation[0],0+robotLocation[1],robotLocation[2]]
        return robotLocation
    
    # chassis movement [0,0,0] # x, y, yaw
    def moveGeneric(self, vector):
        # print("moving generic...")

        # convert real position to chassis local coordinates
        real_pos = self.computeRobotPositionRelative()
        start_pos = self.interface.chassis.get_joint_positions()
        relative_vector = [vector[0]-real_pos[0],vector[1]-real_pos[1], real_pos[2]]
        end_pos = [start_pos[0]+relative_vector[0], start_pos[1]+relative_vector[1], vector[2]]
        print("start: " + str(start_pos))
        print("end: " + str(end_pos))
        positions = np.linspace(start_pos, end_pos, 5) # start_pos, end_pos, 
        # print("trajectory: " + str(positions))
        trajectory = JointTrajectory(positions=positions)

        self.interface.chassis.follow_trajectory(trajectory)

    
    ### Moving dynamically based on yaw
    # https://www.desmos.com/calculator/2wknuddhgu

    def moveForward(self, step):
        if step < 0:
            # ensure movement is forwards
            step = step * -1

        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2])*step), current_pos[1] + (math.sin(current_pos[2])*step), current_pos[2]])

        ### 
        # self.moveGeneric([step,0,0])

    def moveBackwards(self, step):
        if 0 < step:
            # ensure movement is backwards
            step = step * -1


        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2])*step), current_pos[1] + (math.sin(current_pos[2])*step), current_pos[2]])

        ###
        # self.moveGeneric([step,0,0])

    def moveLeft(self, step):
        if step < 0:
            # ensure movement is left
            step = step * -1

        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2]+(math.pi/2))*step), current_pos[1] + (math.sin(current_pos[2]+(math.pi/2))*step), current_pos[2]])


        ###
        # self.moveGeneric([0,step,0])

    def moveRight(self, step):
        if step < 0:
            # ensure movement is right
            step = step * -1

        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2]-(math.pi/2))*step), current_pos[1] + (math.sin(current_pos[2]-(math.pi/2))*step), current_pos[2]])

        ###
        # self.moveGeneric([0,step,0])

    def shiftYaw(self, step):
        
        # append translation to fifo queue
        current_pos = self.fifoPath[-1]
        self.fifoPath.append([current_pos[0], current_pos[1], current_pos[2]+step])

        ###
        # self.moveGeneric([0,0,step])

    def check_movement_complete(self, target, tolerance):
        current = self.computeRobotPositionRelative()
        # print("robot is at " + str(current) + ", aiming to go " + str(target))

        # check if robot has reached target within a tolerance 
        if np.allclose(current, target, atol=tolerance): 
            return True

    def follow_path_callback(self):
        # print("Local Chassis coordinate: " + str(self.interface.chassis.get_joint_positions()))
        # ensure sim length is below 3000 steps
        # if 3000 <= (self.simulator.get_step_count()-self.stepOffset):
        #     self.done = True # ran out of time
        #     return 
        
        # if there is a movement command in queue
        if (len(self.fifoPath) != 0):
            
            # load command from queue
            target = self.fifoPath[0]
            if (self.check_movement_complete(target, 0.1)): # if target has been reached within 0.1 tolerance

                # print(self.fifoPath)
                # print("pop")
                self.fifoPath.pop(0) # remove element from queue
                # print(self.fifoPath)
                self.moving = False

                if (len(self.fifoPath) != 0): # if another element in queue
                    target = self.fifoPath[0]
                    self.moveGeneric(target) # move to target
                    self.moving = True
                    # self.follow_path_callback() # and then run the loop again
        # else:
            # if not, remove any residual callbacks
            # self.simulator.remove_physics_callback("follow_path_callback")

    def step(self, action):
        current_joint_positions = self.interface.chassis.get_joint_positions()
        self.fifoPath.append(current_joint_positions)
        match action:
            case 0:
                self.moveForward(1)
            case 1:
                self.moveBackwards(0.5)
            case 2:
                self.moveLeft(0.5)
            case 3:
                self.moveRight(0.5)
            case 4:
                self.shiftYaw(0.5)
            case 5:
                self.shiftYaw(-0.5)

    
        


test = environment()
test.main()