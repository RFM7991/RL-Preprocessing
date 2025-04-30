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

# TODO:
# look at reward function
#  evluate the model on a test set of images
# do 20/80 # train/test split
# look for similat results 
# play with steps 
# if needed: add all the YOLO training data, just hold out test set for CNN performance evaluation
# 

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

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
    gamma = 1.0
    tau = 0.005
    actor_lr = 1e-3
    critic_lr = 1e-3
    batch_size = 32
    memory_capacity = 100000
    num_episodes = 20000
    num_experiments = 5
    num_steps = 1
    action_noise_std = 0.1
    initial_noise_std = 0.1
    final_noise_std = 0.05
    rewards = []
    differences = []
    successful_detections = []
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_experiments):
        model_path = "models/YOLO_eye_detector.pt"
        image_folder = "images/pupils_xor_iris"
        detector = ObjectDetectorCNN(model_path)
        env = ImagePreprocessingDQNEnv(detector, image_folder, render=False)

        actor = Actor(input_dim=2, output_dim=2).to(device)
        target_actor = Actor(input_dim=2, output_dim=2).to(device)
        target_actor.load_state_dict(actor.state_dict())

        critic = Critic(input_dim=2, action_dim=2).to(device)
        target_critic = Critic(input_dim=2, action_dim=2).to(device)
        target_critic.load_state_dict(critic.state_dict())

        actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

        replay_buffer = ReplayBuffer(memory_capacity)

        selected_actions = []

        for episode in range(num_episodes):
            env.reset()
            state = env.get_state_vector()
            total_reward = 0
            done = False

            for step in range(num_steps):
                if done:
                    break
                noise_decay = max(0, (num_episodes - episode) / num_episodes)
                action_noise_std = initial_noise_std * noise_decay + final_noise_std * (1 - noise_decay)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = actor(state_tensor).cpu().data.numpy().flatten()
                action += np.random.normal(0, action_noise_std, size=2)
                action = np.clip(action, -1.0, 1.0)

                beta = action[0] * 100  # scale to [0, 100]
                alpha = action[1] * 0.3 + 1.5  # scale back to [0.5, 1.5]
                env.current_beta = beta
                env.current_alpha = alpha
                env.image = env.apply_adjustments(env.original_image, beta, alpha)

                original_detections = env.detector.detect_objects(env.detector.preprocess_image_array(env.original_image))
                adjusted_detections = env.detector.detect_objects(env.detector.preprocess_image_array(env.image))
                reward = env.get_reward(original_detections, adjusted_detections)

                selected_actions.append([beta, alpha])

                next_state = env.get_state_vector()
                
                if reward > 0:
                    done = True

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # print(f"Episode {episode+1}, Step {step+1}, State: {state}, Action: [{beta:.2f}, {alpha:.2f}], Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

                # print(f"Episode {episode+1}, State: {state}, Action: [{beta:.2f}, {alpha:.2f}], Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

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
                past_100_avg_rewards = np.mean(env.all_rewards[-100:]) if len(env.all_rewards) >= 100 else np.mean(env.all_rewards)
                past_100_avg_differences = np.mean(env.all_differences[-100:]) if len(env.all_differences) >= 100 else np.mean(env.all_differences)
                past_100_avg_successful_detections = np.mean(env.successful_detections[-100:]) if len(env.successful_detections) >= 100 else np.mean(env.successful_detections)
                print(f"Experiment {i+1}/{num_experiments}, Episode {episode+1}/{num_episodes}, Total Avg. Reward: {np.mean(env.all_rewards):.2f}, 100 Eps Avg. Reward: {past_100_avg_rewards:.2f}, Total Avg. Diff: {np.mean(env.all_differences):.2f}, 100 Eps Diff: {past_100_avg_differences:.2f}, Successful Detections: {np.mean(env.successful_detections):.2f}, 100 Eps Successful Detections: {past_100_avg_successful_detections:.2f}")
                env.plot_rewards(save=True, model_type="DDPG")
                env.plot_differences(save=True, model_type="DDPG")
                env.plot_successful_detections(save=True, model_type="DDPG")

                # plot actions 
                actions_array = np.array(selected_actions)
                episodes = np.arange(len(actions_array))

                fig, ax1 = plt.subplots(figsize=(20, 10))

                # Brightness (Beta) on primary y-axis
                ax1.plot(episodes, actions_array[:, 0], color='tab:blue', label='Brightness Adjustment')
                ax1.set_ylabel('Brightness', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')

                # Contrast (Alpha) on secondary y-axis
                ax2 = ax1.twinx()
                ax2.plot(episodes, actions_array[:, 1], color='tab:orange', label='Contrast Adjustment')
                ax2.set_ylabel('Contrast', color='tab:orange')
                ax2.tick_params(axis='y', labelcolor='tab:orange')

                plt.title("Actions Over Time")
                fig.tight_layout()
                plt.savefig(f"output/DDPG/actions_plot_experiment_{i+1}.png")
                plt.close()

        rewards.append(env.all_rewards)
        differences.append(env.all_differences)
        successful_detections.append(env.successful_detections)
        print(f"Experiment {i+1}/{num_experiments}, Avg. Reward: {np.mean(env.all_rewards):.2f}, Avg. Difference: {np.mean(env.all_differences):.2f}, Successful Detections: {np.mean(env.successful_detections)}")

        avg_rewards = np.mean(rewards, axis=0)
        avg_differences = np.mean(differences, axis=0)
        env.plot_avg_rewards(avg_rewards, save=True, model_type="DDPG")
        env.plot_avg_differences(avg_differences, save=True, model_type="DDPG")
        env.plot_avg_successful_detections(np.mean(successful_detections, axis=0), save=True, model_type="DDPG")
        print(f"Average Reward over {num_experiments} experiments: {np.mean(avg_rewards):.2f}", f"Average Difference: {np.mean(avg_differences):.2f}, Successful Detections: {np.mean(successful_detections):.2f}")

    print("DDPG training complete.")
    # save the model
    torch.save(actor.state_dict(), "models/DDPG_actor.pth")
    torch.save(critic.state_dict(), "models/DDPG_critic.pth")
    print("Models saved.")
