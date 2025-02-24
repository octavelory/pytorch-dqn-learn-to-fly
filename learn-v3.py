#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

# ---------------------------
# Hyperparamètres
# ---------------------------
GAMMA = 0.99        # Facteur de discount
LR = 1e-3           # Taux d'apprentissage
BATCH_SIZE = 64     # Taille de mini-lot pour l'entraînement
MEM_CAPACITY = 10000
EPS_START = 1.0     # Epsilon initial (exploration)
EPS_END = 0.01      # Epsilon minimal
EPS_DECAY = 0.995   # Facteur de décroissance d'epsilon
MAX_STEPS = 500     # Nombre de steps max par épisode
TARGET_UPDATE = 200 # Fréquence de mise à jour du réseau-cible (en steps)

class PlaneEnv:
    """
    Environnement simplifié : un cube “bleu” devant décoller et maintenir
    une altitude dans [min_alt, max_alt].
    Pénalité si l’avion sort de la zone ou se crashe (z < 0.01).
    Nouveau : Step reward proportionnel à l’altitude lors du vol pour encourager
    à monter plus haut dans la zone autorisée.
    """

    def __init__(self, gui=False):
        self.gui = gui
        # Altitudes limites
        self.min_alt = 2.0
        self.max_alt = 20.0

        # Connexion PyBullet
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # IDs
        self.plane_id = None
        self.ground_id = None

        # État interne
        self.takeoff = False
        self.throttle = 0.0
        self.pitch_cmd = 0.0
        self.step_count = 0

        # Actions discrètes
        # 0: rien, 1: accélère, 2: réduit accél, 3: pitch up, 4: pitch down
        self.action_size = 5

        # Paramètres “physiques”
        self.lift_coeff = 5.0
        self.max_force = 100.0

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Sol
        self.ground_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.ground_id, -1, lateralFriction=0.0)

        # Cube bleu
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05],
            rgbaColor=[0, 0, 1, 1]  # Bleu
        )
        start_pos = [0, 0, 0.05]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.plane_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos,
            baseOrientation=start_orn
        )

        # Paramétrage inertie & friction
        p.changeDynamics(self.plane_id, -1,
                         lateralFriction=0.0,
                         angularDamping=0.1,
                         linearDamping=0.0,
                         localInertiaDiagonal=[0.1, 5.0, 0.1])

        # Réinitialise
        self.takeoff = False
        self.throttle = 0.0
        self.pitch_cmd = 0.0
        self.step_count = 0

        return self.get_observation()

    def step(self, action):
        self.step_count += 1

        # Mise à jour des commandes
        if action == 1:  # Throttle up
            self.throttle += 0.1
        elif action == 2:  # Throttle down
            self.throttle -= 0.1
        elif action == 3:  # Pitch up
            self.pitch_cmd += 0.02
        elif action == 4:  # Pitch down
            self.pitch_cmd -= 0.02

        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.pitch_cmd = np.clip(self.pitch_cmd, -0.15, 0.15)

        # Poussée
        force = self.throttle * self.max_force
        p.applyExternalForce(self.plane_id, -1, [force, 0, 0], [0, 0, 0], p.WORLD_FRAME)

        # Couple de tangage (repère local)
        torque = [0, self.pitch_cmd * 1.5, 0]
        p.applyExternalTorque(self.plane_id, -1, torque, p.LINK_FRAME)

        # Portance
        pos, orn = p.getBasePositionAndOrientation(self.plane_id)
        vel_lin, vel_ang = p.getBaseVelocity(self.plane_id)
        pitch = p.getEulerFromQuaternion(orn)[1]
        horizontal_vel = np.sqrt(vel_lin[0]**2 + vel_lin[1]**2)
        lift = self.lift_coeff * (horizontal_vel**2) * np.sin(pitch)
        if lift > 0:
            p.applyExternalForce(self.plane_id, -1, [0, 0, lift], [0, 0, 0], p.WORLD_FRAME)

        # Avance la simulation
        p.stepSimulation()
        if self.gui:
            time.sleep(0.01)

        # Observations
        obs = self.get_observation()
        z = obs[0]
        done = False
        reward = 0.0

        # Mise à jour du statut "takeoff"
        if not self.takeoff and z > self.min_alt:
            self.takeoff = True

        # Calcul de la récompense
        if self.takeoff:
            # Hors zone
            if z < self.min_alt or z > self.max_alt:
                reward = -10.0
                done = True
            else:
                # *Modification* : on valorise l'altitude dans [min_alt, max_alt]
                # => plus z est grand, plus la récompense est importante
                normalized_alt = (z - self.min_alt) / (self.max_alt - self.min_alt)
                reward = 1.0 + 1.0 * normalized_alt
        else:
            # Reward shaping avant décollage
            shaping_vel = min(horizontal_vel, 5.0) / 5.0
            shaping_alt = max(z - 0.05, 0.0)
            reward = 0.2 * shaping_vel + 0.05 * shaping_alt

        # Crash si on s'enfonce sous le sol
        if z < 0.01:
            reward = -10.0
            done = True

        # Limite de pas
        if self.step_count >= MAX_STEPS:
            done = True

        return obs, reward, done, {}

    def get_observation(self):
        """
        Observations :
         - z: altitude
         - pitch: angle de tangage
         - vz: vitesse verticale
         - pitch_rate: vitesse angulaire en tangage
         - throttle: poussée normalisée
        """
        pos, orn = p.getBasePositionAndOrientation(self.plane_id)
        vel_lin, vel_ang = p.getBaseVelocity(self.plane_id)

        z = pos[2]
        pitch = p.getEulerFromQuaternion(orn)[1]
        vz = vel_lin[2]
        pitch_rate = vel_ang[1]
        return np.array([z, pitch, vz, pitch_rate, self.throttle], dtype=np.float32)

    def close(self):
        p.disconnect()

# ---------------------------
# Réseau DQN
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# Mémoire de Replay
# ---------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ---------------------------
# Entraînement DQN
# ---------------------------
def train_dqn(env, episodes, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = len(env.reset())
    action_size = env.action_size

    # Réseaux
    policy_net = QNetwork(state_size, action_size).to(device)
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEM_CAPACITY)

    epsilon = EPS_START
    global_steps = 0

    for ep in range(episodes):
        state = env.reset()
        ep_max_alt = state[0]  # pour suivi
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

        episode_reward = 0.0
        done = False

        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_t)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, done, _ = env.step(action)
            ep_max_alt = max(ep_max_alt, next_state[0])  # suivi altitude max

            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward_t = torch.FloatTensor([reward]).to(device)
            done_t = torch.FloatTensor([float(done)]).to(device)

            # Stockage
            memory.push((state_t, action, reward_t, next_state_t, done_t))

            state_t = next_state_t
            episode_reward += reward
            global_steps += 1

            # Entraînement si assez d'échantillons
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)

                state_batch = torch.cat([t[0] for t in transitions]).to(device)
                action_batch = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1).to(device)
                reward_batch = torch.cat([t[2] for t in transitions]).to(device)
                next_state_batch = torch.cat([t[3] for t in transitions]).to(device)
                done_batch = torch.cat([t[4] for t in transitions]).to(device)

                # Q(s,a)
                q_values = policy_net(state_batch).gather(1, action_batch)

                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).max(1)[0]
                target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

                loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update du target_net
            if global_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        print(f"Episode {ep}/{episodes} - Reward: {episode_reward:.2f} "
              f"- Eps: {epsilon:.3f} - Max Alt: {ep_max_alt:.2f}")

    torch.save(policy_net.state_dict(), save_path)
    env.close()
    print(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")

# ---------------------------
# Démonstration
# ---------------------------
def demo(env, load_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = len(env.reset())
    action_size = env.action_size

    policy_net = QNetwork(state_size, action_size).to(device)
    if os.path.exists(load_path):
        policy_net.load_state_dict(torch.load(load_path, map_location=device))
        print("Modèle chargé depuis", load_path)
    else:
        print("Aucun modèle trouvé. Démo aléatoire.")

    policy_net.eval()

    for i in range(5):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_t)
                action = q_values.argmax(dim=1).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        print(f"Episode Demo {i+1} - Récompense = {episode_reward:.2f}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "demo"], default="demo",
                        help="Mode d'exécution : 'train' ou 'demo'")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--save-path", type=str, default="plane_model.pth",
                        help="Chemin de sauvegarde du modèle entraîné")
    parser.add_argument("--load-path", type=str, default="plane_model.pth",
                        help="Chemin de chargement du modèle (mode demo)")
    args = parser.parse_args()

    if args.mode == "train":
        env = PlaneEnv(gui=False)
        train_dqn(env, args.episodes, args.save_path)
    else:
        env = PlaneEnv(gui=True)
        demo(env, args.load_path)

if __name__ == "__main__":
    main()