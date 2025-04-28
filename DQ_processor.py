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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
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
        # Normalize state to [0, 1]
        return np.array(self.state) / self.num_bins


if __name__ == "__main__":
    num_bins = 10
    num_actions = num_bins * num_bins
    gamma = 0.99
    epsilon_start = 0.2
    epsilon_decay = 0.999
    epsilon_min = 0.01
    batch_size = 32
    target_update_freq = 10
    memory_capacity = 10000
    num_episodes = 1000
    num_experiments = 10
    rewards = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_experiments):
        epsilon = epsilon_start
        model_path = "models/YOLO_eye_detector.pt"
        image_folder = "images/no_pupil"
        detector = ObjectDetectorCNN(model_path)
        env = ImagePreprocessingDQNEnv(detector, image_folder, render=False, num_bins=num_bins)

        policy_net = DQN(input_dim=2, output_dim=num_actions).to(device)
        target_net = DQN(input_dim=2, output_dim=num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters())
        replay_buffer = ReplayBuffer(memory_capacity)

        for episode in range(num_episodes):
            env.reset()
            state = env.get_state_vector()
            total_reward = 0
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action_index = np.random.randint(num_actions)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = policy_net(state_tensor)
                        action_index = q_values.max(1)[1].item()

                next_state, reward, done = env.step(action_index)
                next_state_vector = np.array(next_state) / num_bins

                replay_buffer.push(state, action_index, reward, next_state_vector, done)
                state = next_state_vector
                total_reward += reward

                if len(replay_buffer) > batch_size:
                    states, actions, rewards_batch, next_states, dones = replay_buffer.sample(batch_size)

                    states_tensor = torch.FloatTensor(states).to(device)
                    actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards_tensor = torch.FloatTensor(rewards_batch).to(device)
                    next_states_tensor = torch.FloatTensor(next_states).to(device)
                    dones_tensor = torch.FloatTensor(dones).to(device)

                    q_values = policy_net(states_tensor).gather(1, actions_tensor).squeeze()
                    next_q_values = target_net(next_states_tensor).max(1)[0]
                    expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                    loss = nn.MSELoss()(q_values, expected_q_values.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            env.all_rewards.append(total_reward)
            if episode % 100 == 0:
                past_100_avg_rewards = np.mean(env.all_rewards[-100]) if len(env.all_rewards) >= 100 else np.mean(env.all_rewards)
                print(f"Experiment {i+1}/{num_experiments}, Episode {episode+1}/{num_episodes}, Total Avg. Reward: {np.mean(env.all_rewards):.2f}, 100 eps Avg. Reward: {past_100_avg_rewards:.2f}, Epsilon: {epsilon:.3f}")
                env.plot_rewards(save=True)

        rewards.append(env.all_rewards)
        print(f"Experiment {i+1}/{num_experiments}, Avg. Reward: {np.mean(env.all_rewards):.2f}")

    avg_rewards = np.mean(rewards, axis=0)
    print(f"Average Reward over {num_experiments} experiments: {np.mean(avg_rewards):.2f}")
    env.plot_avg_rewards(avg_rewards, save=True)
    print("DQN training complete.")
