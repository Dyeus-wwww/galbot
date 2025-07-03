from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig
import numpy as np
import time, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os

# ... (your existing imports remain the same) ...

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps)
                       
# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.tau = 1e-3
        self.lr = 1e-3
        self.update_every = 4
        self.batch_size = 64
        self.buffer_size = 10000
        
        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", 
                            field_names=["state", "action", "reward", "next_state", "done"])
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        # Learn every UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)

    def learn(self):
        experiences = self.sample()
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.qnetwork_local.load_state_dict(torch.load(filename))
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

class BotTest1():
    # ... (your existing methods remain the same) ...
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

    def init_pose_done(self):
            leg = [0.43, 1.48, 1.07, 0.0]
            left_arm = [
            0.058147381991147995,
            1.4785659313201904,
            -0.0999724417924881,
            -2.097979784011841,
            1.3999720811843872,
            -0.009971064515411854,
            1.0999830961227417,
            ]
            right_arm = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417,
            ]
            current_positions_ra = self.interface.right_arm.get_joint_positions()
            current_positions_la = self.interface.left_arm.get_joint_positions()
            current_positions_leg = self.interface.leg.get_joint_positions()
            return ((np.allclose(current_positions_leg, leg, atol=0.1) and
                    np.allclose(current_positions_la, left_arm, atol=0.1) and
                    np.allclose(current_positions_ra, right_arm, atol=0.1)))


        
        

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

        self._init_scene()

        # Initialize the simulator
        self.simulator.initialize()

        # self.robot = self.simulator.get_robot("/World/Galbot")

    def _init_scene(self):
        """
        Initialize the scene with tables, closet, and cubes.
        """
        pass

    def init_point(self):
        self.start_point = [random.randint(0,10),random.randint(0,10),0]
        self.end_point = [random.randint(0,10),random.randint(0,10),0]
        self.sim_end_pt = [(self.end_point[0]-self.start_point[0]),(self.end_point[1]-self.start_point[1]),0]
        # self.fifoPath.append(self.sim_end_pt)
        

    def setup_scence(self):
        print("scence")

    # def observe(self):



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

        if (not self.init_pose_done()):
            # print(self.init_pose_done())
            return 
        
        # if there is a movement command in queue
        if (len(self.fifoPath) != 0):
            
            # load command from queue
            target = self.fifoPath[0]
            # print(self.check_movement_complete(target, 0.1) if self.check_movement_complete(target, 0.1) else "") 
            if (self.check_movement_complete(target, 0.1)): # if target has been reached within 0.1 tolerance
                
                # print("pop")
                
                self.fifoPath.pop(0) # remove element from queue
                # print(self.fifoPath)
                # print(x)
                self.moving = False
                if (len(self.fifoPath) != 0): # if another element in queue
                    target = self.fifoPath[0]
                    self.moveGeneric(target) # move to target
                    self.moving = True
                    # self.follow_path_callback() # and then run the loop again

    def moveGeneric(self,target):
        self._move_joints_to_target(self.interface.chassis,target)

    def moveForwards(self,step):
        if (len(self.fifoPath) == 0):
            current_positions = self.interface.chassis.get_joint_positions()
            self.fifoPath.append(current_positions)
        current = list(self.fifoPath[-1])
        current[0] += step
        self.fifoPath.append(current)

    def moveBackwards(self,step):
        self.moveForwards((-abs(step)))

    def moveLeft(self,step):
        if (len(self.fifoPath) == 0):
            current_positions = self.interface.chassis.get_joint_positions()
            self.fifoPath.append(current_positions)
        current = list(self.fifoPath[-1])
        current[1] += step
        self.fifoPath.append(current)

    def moveRight(self,step):
        self.moveLeft((-abs(step)))

    def pitchYawLeft(self,step):
        if (len(self.fifoPath) == 0):
            current_positions = self.interface.chassis.get_joint_positions()
            self.fifoPath.append(current_positions)
        new_target = list(self.fifoPath[-1])
        new_target[2] += step
        self.fifoPath.append(new_target)


    def pitchYawRight(self,step):
        self.pitchYawLeft((-abs(step)))

    def _get_state(self):
        """Get current state for RL agent"""
        # Example state: chassis positions + distance to target
        chassis_pos = np.array(self.interface.chassis.get_joint_positions())
        target_pos = np.array(self.end_point)
        distance = np.linalg.norm(chassis_pos[:2] - target_pos[:2])
        state = np.concatenate([chassis_pos, [distance]])
        return state

    def _calculate_reward(self):
        """Calculate reward based on current state"""
        chassis_pos = np.array(self.interface.chassis.get_joint_positions())
        target_pos = np.array(self.sim_end_pt)
        
        # Distance to target
        distance = np.linalg.norm(chassis_pos[:2] - target_pos[:2])
        
        # Base reward is negative distance (encourage getting closer)
        reward = -distance * 0.1
        
        # Large positive reward for reaching target
        if distance < 0.5:
            reward += 100
            self.target_reached = True
            
        # Negative reward for taking too long
        reward -= 0.01
        
        return reward

    def _reset_episode(self):
        """Reset for a new RL episode"""
        self.target_reached = False
        self.step_count = 0
        self.total_reward = 0
        self.last_state = self._get_state()
        self.fifoPath = []
        self.interface.chassis.set_joint_positions([0, 0, 0])
        self.init_point()

    def main(self):
        self.setup_sim()
        self._setup_interface()
        self._init_pose()
        
        self.simulator.add_physics_callback("follow_path_callback", self.follow_path_callback)
        self.moving = False
        self.fifoPath = []
        self.init_point()
        
        # RL setup
        state_size = 4  # 3 chassis positions + distance to target
        action_size = 6  # 6 movement actions
        self.agent = DQNAgent(state_size, action_size)
        self.agent.load("dqn_checkpoint.pth")  # Load if exists
        
        self.target_reached = False
        self.step_count = 0
        self.max_steps = 1000
        self.last_state = self._get_state()
        self.total_reward = 0
        
        # Start simulation
        self.simulator.step()
        
        # Main simulation loop
        while True:
            self.simulator.step()
            # print((not self.target_reached))
            # RL logic - only when not moving and path is clear
            if ((not self.moving) and (len(self.fifoPath) == 0) and (not self.target_reached)):
                print(self.sim_end_pt)
                # Get current state
                state = self._get_state()
                print(state)
                # Choose action
                action = self.agent.act(state)
                print(action)
                # Execute action
                if action == 0:
                    self.moveForwards(0.5)
                elif action == 1:
                    self.moveBackwards(0.5)
                elif action == 2:
                    self.moveLeft(0.5)
                elif action == 3:
                    self.moveRight(0.5)
                elif action == 4:
                    self.pitchYawLeft(0.1)
                elif action == 5:
                    self.pitchYawRight(0.1)
                print(self.fifoPath)
                
                # Get next state and reward
                next_state = self._get_state()
                reward = self._calculate_reward()
                done = self.target_reached or (self.step_count >= self.max_steps)
                
                # Save experience and learn
                self.agent.step(state, action, reward, next_state, done)
                
                # Update tracking variables
                self.total_reward += reward
                self.step_count += 1
                self.last_state = next_state
                
                # Reset if episode ended
                if done:
                    print(f"Episode ended! Steps: {self.step_count}, Reward: {self.total_reward:.2f}")
                    self.agent.save("dqn_checkpoint.pth")
                    self._reset_episode()

if __name__ == "__main__":
    test = BotTest1()
    test.main()