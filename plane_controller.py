import pybullet as p
import pybullet_data
import time
import math
import argparse
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

FIX_Y_AXIS = True

def create_plane():
    
    fuselage_half_extents = [0.5, 0.1, 0.1]
    fuselage_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=fuselage_half_extents)
    fuselage_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=fuselage_half_extents, rgbaColor=[1, 0, 0, 1])

    wing_half_extents = [0.1, 0.3, 0.02]
    wing_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=wing_half_extents)
    wing_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=wing_half_extents, rgbaColor=[0, 1, 0, 1])

    tail_half_extents = [0.05, 0.1, 0.01]
    tail_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=tail_half_extents)
    tail_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=tail_half_extents, rgbaColor=[0, 0, 1, 1])

    base_mass = 1
    base_position = [0, 0, 0.1]
    base_orientation = p.getQuaternionFromEuler([0, 0, 0])

    linkMasses = [0.2, 0.2, 0.1]
    linkCollisionShapeIndices = [wing_collision, wing_collision, tail_collision]
    linkVisualShapeIndices = [wing_visual, wing_visual, tail_visual]
    linkPositions = [
        [0, 0.4, 0],
        [0, -0.4, 0],   
        [-0.55, 0, 0]   
    ]
    linkOrientations = [p.getQuaternionFromEuler([0, 0, 0]) for _ in range(3)]
    linkInertialFramePositions = [[0, 0, 0] for _ in range(3)]
    linkInertialFrameOrientations = [p.getQuaternionFromEuler([0, 0, 0]) for _ in range(3)]
    linkParentIndices = [0, 0, 0]
    linkJointTypes = [p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED]
    linkJointAxis = [[0, 0, 0] for _ in range(3)]

    plane_id = p.createMultiBody(
        baseMass=base_mass,
        baseCollisionShapeIndex=fuselage_collision,
        baseVisualShapeIndex=fuselage_visual,
        basePosition=base_position,
        baseOrientation=base_orientation,
        linkMasses=linkMasses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=linkParentIndices,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis,
    )
    return plane_id

class PlaneEnv:
    def __init__(self, render=False, action_repeat=10):
        
        self.render = render
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        
        self.dt = 1.0 / 240.0         
        self.action_repeat = action_repeat  

        
        self.max_thrust = 50            
        self.thrust_increment = 0.5     
        self.current_thrust = 0         

        
        self.lift_gain = 0.15           
        self.drag_gain = 0.15           
        self.thrust_threshold = 0.01    
        self.velocity_threshold = 0.01  

        
        self.Kp = 10
        self.Kd = 2

        
        self.fix_y_axis = FIX_Y_AXIS
        if self.fix_y_axis:
            self.kp_y = 100       
            self.kd_y = 20        
            self.Kp_yaw = 50      
            self.Kd_yaw = 10

        
        self.target_altitude = 5.0     
        self.max_steps = 2500           

        self.step_count = 0
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")  
        self.plane_id = create_plane()
        self.current_thrust = 0
        self.step_count = 0
        state = self._get_state()
        return state

    def _get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.plane_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.plane_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        
        rot_matrix = p.getMatrixFromQuaternion(orn)
        side = [rot_matrix[3], rot_matrix[4], rot_matrix[5]]
        pitch_rate = sum([ang_vel[i] * side[i] for i in range(3)])
        
        state = np.array([pos[2], lin_vel[2], pitch, pitch_rate], dtype=np.float32)
        return state

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self.action_repeat):
            self.step_count += 1
            
            if action == 1:
                self.current_thrust += self.thrust_increment
            else:
                self.current_thrust *= 0.70
            self.current_thrust = max(-self.max_thrust, min(self.max_thrust, self.current_thrust))

            pos, orn = p.getBasePositionAndOrientation(self.plane_id)
            lin_vel, ang_vel = p.getBaseVelocity(self.plane_id)
            rot_matrix = p.getMatrixFromQuaternion(orn)
            
            forward = [rot_matrix[0], rot_matrix[1], rot_matrix[2]]
            side = [rot_matrix[3], rot_matrix[4], rot_matrix[5]]
            forward_velocity = sum([lin_vel[i] * forward[i] for i in range(3)])
            forward_velocity_abs = abs(forward_velocity)
            
            thrust_force = [self.current_thrust * forward[i] for i in range(3)]
            
            if self.current_thrust > self.thrust_threshold and forward_velocity > self.velocity_threshold:
                effective_lift_gain = self.lift_gain * ((self.current_thrust - self.thrust_threshold) / (self.max_thrust - self.thrust_threshold))
            else:
                effective_lift_gain = 0
            lift = effective_lift_gain * (forward_velocity_abs ** 2)
            lift_force = [0, 0, lift]
            
            drag_force = [-self.drag_gain * v for v in lin_vel]
            
            total_force = [thrust_force[i] + lift_force[i] + drag_force[i] for i in range(3)]

            
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            pitch_rate = sum([ang_vel[i] * side[i] for i in range(3)])
            correction = -self.Kp * pitch - self.Kd * pitch_rate
            corrective_torque = [correction * side[i] for i in range(3)]

            
            if self.fix_y_axis:
                corrective_force_y = -self.kp_y * pos[1] - self.kd_y * lin_vel[1]
                total_force[1] = corrective_force_y  
                
                _, _, curr_yaw = p.getEulerFromQuaternion(orn)
                yaw_correction = -self.Kp_yaw * curr_yaw - self.Kd_yaw * ang_vel[2]
                corrective_torque[2] += yaw_correction

            
            p.applyExternalForce(self.plane_id, -1, total_force, pos, p.WORLD_FRAME)
            p.applyExternalTorque(self.plane_id, -1, corrective_torque, p.WORLD_FRAME)

            p.stepSimulation()
            if self.render:
                time.sleep(self.dt)

            
            if pos[2] < 0:
                done = True
                total_reward += -100  
                break

        next_state = self._get_state()
        
        
        altitude = next_state[0]
        vertical_vel = next_state[1]
        pitch = next_state[2]
        reward = -abs(altitude - self.target_altitude) - 0.1 * (vertical_vel**2) - 0.1 * (pitch**2)
        total_reward += reward

        if self.step_count >= self.max_steps:
            done = True

        return next_state, total_reward, done, {}




class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), action, reward, np.stack(next_state), done
    def __len__(self):
        return len(self.buffer)




def train_dqn(env, num_episodes, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4   
    output_dim = 2  
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(50000)
    batch_size = 64
    gamma = 0.99
    target_update_frequency = 1000  
    total_steps = 0

    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 5000  

    print("entrainement démarré")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            total_steps += 1
            
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * total_steps / epsilon_decay)
            if random.random() < epsilon:
                action = random.randrange(output_dim)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_values = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, target_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{num_episodes}\tTotal reward: {episode_reward:.2f}\tepsilon: {epsilon:.3f}")
    torch.save(policy_net.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")




def demo_dqn(env, load_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4
    output_dim = 2
    policy_net = DQN(input_dim, output_dim).to(device)
    policy_net.load_state_dict(torch.load(load_path, map_location=device))
    policy_net.eval()
    print("Mode démonstration - Appuyez sur CTRL+C pour quitter.")
    state = env.reset()
    try:
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.max(1)[1].item()
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()
    except KeyboardInterrupt:
        print("stopping demo.")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'demo'], required=True, help="Mode d'exécution : train ou demo")
    parser.add_argument('--load-path', type=str, help="Chemin du modèle à charger (pour demo)")
    parser.add_argument('--save-path', type=str, help="Chemin de sauvegarde du modèle (pour train)")
    parser.add_argument('--episodes', type=int, default=200, help="Nombre d'épisodes d'entraînement")
    args = parser.parse_args()

    if args.mode == 'demo':
        if args.load_path is None:
            parser.error("no load path provided.")
        env = PlaneEnv(render=True)
        demo_dqn(env, args.load_path)
    elif args.mode == 'train':
        if args.save_path is None:
            parser.error("no save path provided.")
        env = PlaneEnv(render=False)
        train_dqn(env, args.episodes, args.save_path)

if __name__ == "__main__":
    main()