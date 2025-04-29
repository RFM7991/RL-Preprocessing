import os
import numpy as np
import cv2
import random
from objectDetectorCNN import ObjectDetectorCNN
from Q_processor import ImagePreprocessingQEnv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ImagePreprocessingDQNEnv(ImagePreprocessingQEnv):
    def get_state_vector(self):
        brightness, contrast = self.compute_image_stats(self.image)
        return np.array([brightness, contrast])

if __name__ == "__main__":
    gamma = 0.99
    tau = 0.005
    actor_lr = 1e-4
    critic_lr = 1e-3
    batch_size = 8
    memory_capacity = 100000
    num_episodes = 20000
    num_experiments = 1
    action_noise_std = 0.1
    rewards = []
    differences = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_experiments):
        model_path = "models/YOLO_eye_detector.pt"
        image_folder = "images/no_pupils"
        detector = ObjectDetectorCNN(model_path)
        env = ImagePreprocessingDQNEnv(detector, image_folder, render=True, num_bins=10)

        actor = Actor(input_dim=2, output_dim=2).to(device)
        target_actor = Actor(input_dim=2, output_dim=2).to(device)
        target_actor.load_state_dict(actor.state_dict())

        critic = Critic(input_dim=2, action_dim=2).to(device)
        target_critic = Critic(input_dim=2, action_dim=2).to(device)
        target_critic.load_state_dict(critic.state_dict())

        actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

        replay_buffer = ReplayBuffer(memory_capacity)

        for episode in range(num_episodes):
            env.reset()
            state = env.get_state_vector()
            total_reward = 0
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = actor(state_tensor).cpu().data.numpy().flatten()
                action += np.random.normal(0, action_noise_std, size=2)
                action = np.clip(action, -1.0, 1.0)

                beta = action[0] * 100  # scale back to [-100, 100]
                alpha = 1.0 + action[1] * 0.5  # scale back to [0.5, 1.5]

                env.current_beta = beta
                env.current_alpha = alpha
                env.image = env.apply_adjustments(env.original_image, beta, alpha)

                original_detections = env.detector.detect_objects(env.detector.preprocess_image_array(env.original_image))
                adjusted_detections = env.detector.detect_objects(env.detector.preprocess_image_array(env.image))
                reward = env.get_reward(original_detections, adjusted_detections)

                next_state = env.get_state_vector()
                done = True

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(replay_buffer) > batch_size:
                    states, actions, rewards_batch, next_states, dones = replay_buffer.sample(batch_size)

                    states_tensor = torch.FloatTensor(states).to(device)
                    actions_tensor = torch.FloatTensor(actions).to(device)
                    rewards_tensor = torch.FloatTensor(rewards_batch).unsqueeze(1).to(device)
                    next_states_tensor = torch.FloatTensor(next_states).to(device)
                    dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

                    next_actions = target_actor(next_states_tensor)
                    next_q_values = target_critic(next_states_tensor, next_actions)
                    expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                    q_values = critic(states_tensor, actions_tensor)
                    critic_loss = nn.MSELoss()(q_values, expected_q_values.detach())

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    actor_loss = -critic(states_tensor, actor(states_tensor)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            env.all_rewards.append(total_reward)
            if episode % 100 == 0:
                past_100_avg_rewards = np.mean(env.all_rewards[-100]) if len(env.all_rewards) >= 100 else np.mean(env.all_rewards)
                past_100_avg_differences = np.mean(env.all_differences[-100]) if len(env.all_differences) >= 100 else np.mean(env.all_differences)
                print(f"Experiment {i+1}/{num_experiments}, Episode {episode+1}/{num_episodes}, Total Avg. Reward: {np.mean(env.all_rewards):.2f}, 100 Eps Avg. Reward: {past_100_avg_rewards:.2f}, Total Avg. Diff: {np.mean(env.all_differences):.2f}, 100 Eps Diff: {past_100_avg_differences:.2f}")
                env.plot_rewards(save=True, model_type="DDPG")
                env.plot_differences(save=True, model_type="DDPG")

        rewards.append(env.all_rewards)
        differences.append(env.all_differences)
        print(f"Experiment {i+1}/{num_experiments}, Avg. Reward: {np.mean(env.all_rewards):.2f}, Avg. Difference: {np.mean(env.all_differences):.2f}")

    avg_rewards = np.mean(rewards, axis=0)
    avg_differences = np.mean(differences, axis=0)
    env.plot_avg_rewards(avg_rewards, save=True, model_type="DDPG")
    env.plot_avg_differences(avg_differences, save=True, model_type="DDPG")
    print(f"Average Reward over {num_experiments} experiments: {np.mean(avg_rewards):.2f}", f"Average Difference: {np.mean(avg_differences):.2f}")

    print("DDPG training complete.")
