import os
import numpy as np
import cv2
import random
from objectDetectorCNN import ObjectDetectorCNN
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_bins = 20
    num_actions = num_bins * num_bins
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.2
    epsilon_decay = 0.9999
    epsilon_min = 0.01
    num_episodes = 1000
    num_experiments = 10
    rewards = []

    for i in range(num_experiments):
        model_path = "models/YOLO_eye_detector.pt"
        image_folder = "images/no_pupils"
        q_table = np.zeros((num_bins, num_bins, num_actions))
        detector = ObjectDetectorCNN(model_path)
        env = ImagePreprocessingQEnv(detector, image_folder, render=True, num_bins=num_bins)

        for episode in range(num_episodes):
            state = env.reset()
            action_index = np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(q_table[state[0], state[1]])

            next_state, reward, done = env.step(action_index)

            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
            td_error = td_target - q_table[state[0], state[1], action_index]
            q_table[state[0], state[1], action_index] += alpha * td_error

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            if episode % 100 == 0:
                past_100_avg_rewards = np.mean(env.all_rewards[-100]) if len(env.all_rewards) >= 100 else np.mean(env.all_rewards)
                print(f"Episode {episode+1}/{num_episodes}, Total Avg.Reward: {np.mean(env.all_rewards):.2f}, 100 eps Avg. Reward: {past_100_avg_rewards:.2}, Epsilon: {epsilon:.3f}")
                env.plot_rewards(save=True, model_type="Q")
            
        rewards.append(env.all_rewards)
        print(f"Experiment {i+1}/{num_experiments}, Avg. Reward: {np.mean(env.all_rewards):.2f}")

    avg_rewards = np.mean(rewards, axis=0)
    print(f"Average Reward over {num_experiments} experiments: {np.mean(avg_rewards):.2f}")
    env.plot_avg_rewards(avg_rewards, save=True, model_type="Q")
    print("Q-learning training complete.")
