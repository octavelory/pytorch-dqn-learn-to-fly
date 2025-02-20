# learn.py
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
MAX_STEPS = 500     # Nombre de steps max par épisode (augmenté)
TARGET_UPDATE = 20  # Fréquence de mise à jour du réseau-cible

# ---------------------------
# Environnement PyBullet
# ---------------------------
class PlaneEnv:
    """
    Environnement simplifié utilisant PyBullet pour simuler un avion
    assimilé à un cube devant décoller et se maintenir entre deux altitudes.
    Modifications:
    - On ajoute un modèle de portance rudimentaire en fonction de la vitesse horizontale et du pitch.
    - On réduit la friction au sol et sur le cube.
    - On descend l'altitude minimale à 0.5.
    - On augmente MAX_STEPS = 500 pour laisser le temps de décoller.
    """

    def __init__(self, gui=False):
        self.gui = gui
        # Altitudes limites
        self.min_alt = 0.5   # Plus bas qu'avant pour faciliter le décollage
        self.max_alt = 10.0
        
        # Lance PyBullet
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Variables internes
        self.plane_id = None       # ID du cube “avion”
        self.ground_id = None      # ID du sol
        self.takeoff = False       # Avion a-t-il décollé ?
        self.throttle = 0.0        # Commande de poussée (0 à 1)
        self.pitch_cmd = 0.0       # Commande de tangage
        self.step_count = 0

        # Actions discrètes :
        # 0: Nothing,
        # 1: Throttle up,
        # 2: Throttle down,
        # 3: Pitch up,
        # 4: Pitch down
        self.action_size = 5

    def reset(self):
        """
        Remet l’environnement à zéro pour un nouvel épisode.
        """
        # Remise à zéro PyBullet
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Chargement du sol
        self.ground_id = p.loadURDF("plane.urdf")
        # Diminue la friction du sol
        p.changeDynamics(self.ground_id, -1, lateralFriction=0.001)

        # Création d'un cube représentant l'avion
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[1, 0, 0, 1]
        )

        start_pos = [0, 0, 0.2]  # Sur le sol
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.plane_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos,
            baseOrientation=start_orn
        )

        # On réduit également la friction du cube
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.001)

        # Réinitialise les variables internes
        self.takeoff = False
        self.throttle = 0.0
        self.pitch_cmd = 0.0
        self.step_count = 0

        # Observation initiale
        return self.get_observation()

    def step(self, action):
        """
        Exécute une action et simule l’environnement sur 1 "pas de temps".
        Renvoie : obs, reward, done, info
        """
        self.step_count += 1
        # Interprétation de l’action
        if action == 1:
            self.throttle += 0.1
        elif action == 2:
            self.throttle -= 0.1
        elif action == 3:
            self.pitch_cmd += 0.05
        elif action == 4:
            self.pitch_cmd -= 0.05

        # Clamp des commandes
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.pitch_cmd = np.clip(self.pitch_cmd, -0.2, 0.2)

        # Applique la force liée à la poussée
        force = self.throttle * 10.0
        # On applique la poussée dans le repère du monde (simplification)
        p.applyExternalForce(
            self.plane_id, -1, [force, 0, 0], [0, 0, 0], p.WORLD_FRAME
        )

        # Applique un couple pour le tangage (pitch)
        torque = [0, self.pitch_cmd * 2.0, 0]
        p.applyExternalTorque(self.plane_id, -1, torque, p.WORLD_FRAME)

        # ---------------------
        # Modèle de portance rudimentaire
        # ---------------------
        pos, orn = p.getBasePositionAndOrientation(self.plane_id)
        vel_lin, vel_ang = p.getBaseVelocity(self.plane_id)
        # Pitch en radians
        pitch = p.getEulerFromQuaternion(orn)[1]
        # Vitesse horizontale (excluant l'axe Z)
        horizontal_vel = np.sqrt(vel_lin[0] ** 2 + vel_lin[1] ** 2)

        # On définit un coefficient de portance
        # (valeur arbitraire à ajuster pour des tests)
        lift_coeff = 5.0
        # Lift = coeff * v^2 * sin(pitch)
        # pitch > 0 => l'avion pointe vers le haut => plus de lift
        lift = lift_coeff * (horizontal_vel ** 2) * np.sin(pitch)

        # Applique la force verticale (en z) due à la portance
        # On la place au centre du cube, en WORLD_FRAME
        p.applyExternalForce(
            self.plane_id, -1, [0, 0, lift], [0, 0, 0], p.WORLD_FRAME
        )

        # Avance la simulation
        p.stepSimulation()
        time.sleep(0.01 if self.gui else 0.0)

        # Récupère la nouvelle observation
        obs = self.get_observation()
        z = obs[0]  # altitude
        done = False
        reward = 0.0

        # Vérifie si l'avion a décollé
        if (not self.takeoff) and z > self.min_alt:
            self.takeoff = True

        # La pénalité s'applique seulement après le décollage
        if self.takeoff:
            if z < self.min_alt or z > self.max_alt:
                # L'avion sort de sa plage d'altitude
                reward = -10.0
                done = True
            else:
                # Récompense positive à chaque step dans la zone
                reward = 1.0
        else:
            # Tant qu'il n'a pas encore décollé,
            # encourage un peu l'élévation
            reward = 0.1 * max(z - 0.2, 0.0)

        # Si l'avion touche (presque) le sol => crash
        if z < 0.05:
            reward = -10.0
            done = True

        # Limite de pas de temps
        if self.step_count >= MAX_STEPS:
            done = True

        return obs, reward, done, {}

    def get_observation(self):
        """
        Construit un vecteur d’observation à partir de la position, orientation, vitesse, etc.
        Ici, on retourne simplement (z, pitch, vz, pitch_rate, throttle).
        """
        pos, orn = p.getBasePositionAndOrientation(self.plane_id)
        vel_lin, vel_ang = p.getBaseVelocity(self.plane_id)
        # Altitude
        z = pos[2]
        # Pitch (en radians)
        pitch = p.getEulerFromQuaternion(orn)[1]
        # Vitesse verticale
        vz = vel_lin[2]
        # Vitesse de tangage
        pitch_rate = vel_ang[1]
        # On assemble tout dans un tableau
        return np.array([z, pitch, vz, pitch_rate, self.throttle], dtype=np.float32)

    def close(self):
        """
        Déconnecte l'environnement.
        """
        p.disconnect()


# ---------------------------
# Réseau de neurones pour la Q-Learning (DQN)
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
# Mémoire Replay
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
# Boucle d'entraînement DQN
# ---------------------------
def train_dqn(env, episodes, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paramètres d'environnement
    state_size = len(env.reset())  # Taille du vecteur d'observation
    action_size = env.action_size

    # Initialisation du réseau Q et du réseau-cible
    policy_net = QNetwork(state_size, action_size).to(device)
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEM_CAPACITY)

    global_step = 0
    epsilon = EPS_START

    for ep in range(episodes):
        state = env.reset()
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
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward_t = torch.FloatTensor([reward]).to(device)
            done_t = torch.FloatTensor([float(done)]).to(device)

            # Stockage dans la Replay Memory
            memory.push((state_t, action, reward_t, next_state_t, done_t))

            state_t = next_state_t
            episode_reward += reward
            global_step += 1

            # Entraînement par batch depuis la mémoire
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                # Regroupe dans des tenseurs
                state_batch = torch.cat([t[0] for t in transitions]).to(device)
                action_batch = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1).to(device)
                reward_batch = torch.cat([t[2] for t in transitions]).to(device)
                next_state_batch = torch.cat([t[3] for t in transitions]).to(device)
                done_batch = torch.cat([t[4] for t in transitions]).to(device)

                # Q(s,a)
                q_values = policy_net(state_batch).gather(1, action_batch)

                # Q-cible = r + gamma * max(Q') * (1 - done)
                next_q_values = target_net(next_state_batch).max(1)[0].detach()
                target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

                loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Mise à jour du target_net régulièrement
            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Décroissance d'epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        print(f"Episode {ep}/{episodes} - Récompense totale: {episode_reward:.2f} - Eps: {epsilon:.3f}")

    # Sauvegarde du réseau entraîné
    torch.save(policy_net.state_dict(), save_path)
    env.close()
    print("Entraînement terminé. Modèle sauvegardé dans", save_path)


# ---------------------------
# Mode Démonstration
# ---------------------------
def demo(env, load_path):
    """
    Charge le réseau entraîné et montre son comportement dans PyBullet (mode GUI).
    """
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

    for i in range(5):  # On fait 5 épisodes de démonstration
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_t)
                action = q_values.argmax(dim=1).item()
            # Applique l'action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        print(f"Episode demo {i+1} - Récompense : {episode_reward:.2f}")

    env.close()

# ---------------------------
# Point d’entrée principal
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "demo"], default="demo",
                        help="Mode d'exécution : 'train' ou 'demo'")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--save-path", type=str, default="plane_dqn.pth",
                        help="Chemin de sauvegarde du modèle entraîné")
    parser.add_argument("--load-path", type=str, default="plane_dqn.pth",
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