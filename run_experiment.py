import os
import numpy as np
import cv2
import random
from objectDetectorCNN import ObjectDetectorCNN
from image_preprocessing_env import ImagePreprocessingQEnv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import shutil
from DDPG_processor import Actor, Critic, ReplayBuffer
from utils import load_model, plot_and_save_results, split_dataset, print_results_dict, plot_metric_over_episodes

def select_action(actor, state, noise_std=0.0, device=None):
    """
    Selects a continuous action from the actor network, with optional Gaussian noise.
    
    Parameters:
    - actor (nn.Module): Trained actor model.
    - state (np.ndarray): Current environment state.
    - noise_std (float): Standard deviation of Gaussian exploration noise.
    - device (torch.device): Device for inference.

    Returns:
    - action (np.ndarray): Clipped continuous action in range [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor.to(device)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = actor(state_tensor).cpu().data.numpy().flatten()

    if noise_std > 0:
        action += np.random.normal(0, noise_std, size=action.shape)

    return np.clip(action, -1.0, 1.0)

def plot_actions(actions_array, episodes):
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
    plt.savefig(f"output/DDPG/plots/actions_plot.png")
    plt.close()


def evaluate_model(actor, critic, env, num_episodes, num_steps=1):
    total_rewards = []
    total_detections = []
    total_differences = []
    total_steps = []

    for episode in range(num_episodes):
        env.reset(shuffle=False)
        state = env.get_state_vector() 
        total_reward = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor.to(device)
        critic.to(device)
        for step in range(num_steps):
            action = select_action(actor, state, noise_std=0.05, device=device)
            beta = action[0] * 100
            alpha = action[1] * 0.3 + 1.5
            action_input = np.array([action[0], action[1]])

            next_state, reward, done, original_detections, adjusted_detections = env.step(action_input)
            total_reward += reward
            state = next_state

            print(f"Episode {episode+1}, Step {step+1}, State: {next_state}, Action: [{beta:.2f}, {alpha:.2f}], Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if done:
                break
        
        # Log after episode ends (whether via done or max steps)
        total_detections.append(len(adjusted_detections))
        total_differences.append(len(adjusted_detections) - len(original_detections))
        total_steps.append(step + 1)
        total_rewards.append(total_reward)

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    return total_detections, total_differences, total_steps, total_rewards


def run_evaluation(actor_path="models/DDPG_actor.pth",
                   critic_path="models/DDPG_critic.pth",
                   num_steps=1,
                   render=False,
                   model_path="models/YOLO_eye_detector.pt",
                   image_folder="images/test"):

    actor, critic = load_model(actor_path, critic_path)
    detector = ObjectDetectorCNN(model_path)
    env = ImagePreprocessingQEnv(detector, image_folder, render=render)

    num_images = len([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    print(f"Evaluating on {num_images} images...")

    detections, differences, steps, rewards  = evaluate_model(actor, critic, env, num_episodes=num_images, num_steps=num_steps)

    plot_and_save_results(detections, differences, steps, num_images, num_steps)
    print("Evaluation complete.")


if __name__ == "__main__":

    # Load the trained model and run evaluation
    # run_evaluation(actor_path="models/5/DDPG_actor.pth", critic_path="models/5/DDPG_critic.pth", num_steps=5, render=True, model_path="models/YOLO_eye_detector.pt", image_folder="images/test")
    print("Evaluation finished.")  

    gamma = 0.99
    tau = 0.005
    actor_lr = 1e-3
    critic_lr = 1e-3
    batch_size = 32
    memory_capacity = 100000
    num_episodes = 10000
    num_experiments = 1
    action_noise_std = 0.1
    initial_noise_std = 0.1
    final_noise_std = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/YOLO_eye_detector.pt"
    train_images_path = "images/train"
    test_images_path = "images/test"

    # optional set seed 
    # random.seed(21)
    # np.random.seed(21)
    # torch.manual_seed(21)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(21)

    step_grid = [5]
    results_dict = {}

    for num_steps in step_grid:
        if num_steps not in results_dict:
            results_dict[num_steps] = {
                "rewards": [],
                "differences": [],
                "successful_detections": []
            }
        
        total_rewards = []
        total_differences = []
        total_successful_detections = []
        print(f"Running DDPG with {num_steps} steps...")
        for i in range(num_experiments):
            train_images_folder, test_images_folder = Path(train_images_path), Path(test_images_path)
            # split_dataset() get a new split saved as train_shuffle and test_shuffle

            detector = ObjectDetectorCNN(model_path)
            env = ImagePreprocessingQEnv(detector, train_images_folder, render=False)

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
            rewards = []
            differences = []
            successful_detections = []

            for episode in range(num_episodes):
           
                env.reset(shuffle=True) 
                state = env.get_state_vector()
                per_episode_reward = 0
                per_episode_detections = 0
                per_episode_differences = 0
                done = False
                for step in range(num_steps):
                    if done:
                        break

                    noise_decay = max(0, (num_episodes - episode) / num_episodes)
                    action_noise_std = initial_noise_std * noise_decay + final_noise_std * (1 - noise_decay)
                    action = select_action(actor, state, noise_std=action_noise_std, device=device)

                    beta = action[0] * 100  # scale to [0, 100]
                    alpha = action[1] * 0.3 + 1.5  # scale back to [0.5, 1.5]
                    selected_actions.append([beta, alpha])
                    action_input = np.array([action[0], action[1]])
                    next_state, reward, done, original_detections, adjusted_detections = env.step(action_input)

                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    per_episode_reward += reward

                    if done or step == num_steps - 1:    
                        per_episode_detections = len(adjusted_detections)
                        per_episode_differences = len(adjusted_detections) - len(original_detections)


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

                rewards.append(per_episode_reward)
                differences.append(per_episode_differences)
                successful_detections.append(per_episode_detections)

                if episode % 100 == 0:
                    past_100_avg_rewards = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                    past_100_avg_differences = np.mean(differences[-100:]) if len(differences) >= 100 else np.mean(differences)
                    past_100_avg_successful_detections = np.mean(successful_detections[-100:]) if len(successful_detections) >= 100 else np.mean(successful_detections)
                    print(f"Experiment {i+1}/{num_experiments}, Episode {episode+1}/{num_episodes}, Total Avg. Reward: {np.mean(rewards):.2f}, 100 Eps Avg. Reward: {past_100_avg_rewards:.2f}, Total Avg. Diff: {np.mean(differences):.2f}, 100 Eps Diff: {past_100_avg_differences:.2f}, Successful Detections: {np.mean(successful_detections):.2f}, 100 Eps Successful Detections: {past_100_avg_successful_detections:.2f}")
                
                    # save plots 
                    actions_array = np.array(selected_actions)
                    episodes = np.arange(len(actions_array))
                    plot_actions(actions_array, episodes)
                    # env.plot_rewards(save=True, rewards=rewards, model_type="DDPG")
                    # env.plot_differences(save=True, differences=differences, model_type="DDPG")
                    # env.plot_successful_detections(save=True, detections=successful_detections, model_type="DDPG")

            total_rewards.append(rewards)
            total_differences.append(differences)
            total_successful_detections.append(successful_detections)
            print(f"Total Avg. Reward: {np.mean(rewards):.2f}, Total Avg. Diff: {np.mean(differences):.2f}, Successful Detections: {np.mean(successful_detections):.2f}")

            # get averages after every experiment, override previous outputs
            avg_rewards = np.mean(total_rewards, axis=0)
            avg_differences = np.mean(total_differences, axis=0)
            avg_successful_detections = np.mean(total_successful_detections, axis=0)
            env.plot_avg_rewards(avg_rewards, save=True, model_type="DDPG")
            env.plot_avg_differences(avg_differences, save=True, model_type="DDPG")
            env.plot_avg_successful_detections(avg_successful_detections, save=True, model_type="DDPG")

        # save the model
        actor_path = f"models/{num_steps}/DDPG_actor.pth"
        critic_path = f"models/{num_steps}/DDPG_critic.pth"
        os.makedirs(os.path.dirname(actor_path), exist_ok=True)
        os.makedirs(os.path.dirname(critic_path), exist_ok=True)
        
        # Save the model state dictionaries
        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
        print(f"Model saved at {actor_path} and {critic_path}")
        print("Models saved.")
        run_evaluation(actor_path=actor_path, critic_path=critic_path, num_steps=num_steps, model_path=model_path, image_folder=test_images_path)

        # results_dict 
        results_dict[num_steps]["rewards"].append(rewards)
        results_dict[num_steps]["differences"].append(differences)
        results_dict[num_steps]["successful_detections"].append(successful_detections)

        print_results_dict(results_dict)
        plot_metric_over_episodes(results_dict, "rewards", f"output/DDPG/plots/multi_rewards_plot.png", "Reward", "Rewards Over Time")
        plot_metric_over_episodes(results_dict, "differences", f"output/DDPG/plots/multi_differences_plot.png", "Difference in Confidence", "Differences Over Time")
        plot_metric_over_episodes(results_dict, "successful_detections", f"output/DDPG/plots/multi_successful_detections_plot.png", "Successful Detections", "Successful Detections Over Time")
        print("Results saved.")

    print("DDPG training complete.")

