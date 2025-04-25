import os
import gym
import numpy as np
import cv2  
import random
from objectDetectorCNN import ObjectDetectorCNN
from pathlib import Path
import matplotlib.pyplot as plt

class ImagePreprocessingQEnv:
    def __init__(self, detector, image_folder, render=False, num_bins=10):
        self.detector = detector
        self.image_folder = Path(__file__).resolve().parent / image_folder
        self.image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        if not self.image_paths:
            raise ValueError(f"No valid images found in {image_folder}")

        self.current_image_index = 0
        self.original_image = None
        self.image = None
        self.render = render
        self.num_bins = num_bins

        # Discrete action space: [-1, 0, 1] for brightness & contrast adjustments
        self.action_space = [(b, c) for b in [-1, 0, 1] for c in [-1, 0, 1]]
        self.state = (self.num_bins // 2, self.num_bins // 2)  # start from middle

        # Normalized initial brightness & contrast
        self.current_brightness = 0.0
        self.current_contrast = 0.0
        self.all_rewards = []

    def reset(self):
        self.current_image_index = random.randint(0, len(self.image_paths) - 1)
        self.original_image = cv2.imread(self.image_paths[self.current_image_index])
        if self.original_image is None:
            raise ValueError(f"Could not read image: {self.image_paths[self.current_image_index]}")

        self.image = self.original_image.copy()
        self.current_brightness = 0.0
        self.current_contrast = 0.0
        self.state = (self.num_bins // 2, self.num_bins // 2)
        return self.state

    def step(self, action_index):
        brightness_change, contrast_change = self.action_space[action_index]

        # Apply small step adjustments
        delta = 0.1  # Step size for brightness/contrast
        self.current_brightness = np.clip(self.current_brightness + brightness_change * delta, -1.0, 1.0)
        self.current_contrast = np.clip(self.current_contrast + contrast_change * delta, -1.0, 1.0)

        self.image = self.apply_adjustments(self.original_image, self.current_brightness, self.current_contrast)

        img_tensor = self.detector.preprocess_image_array(self.image)
        detections = self.detector.detect_objects(img_tensor)

        # Reward
        if detections:
            reward = sum(det['confidence'] for det in detections)
            if len(detections) > 3:
                reward += (3 - len(detections))
        else:
            reward = -1.0

        if self.render:
            print(f"Reward: {reward:.2f}, Brightness: {self.current_brightness:.2f}, Contrast: {self.current_contrast:.2f}")
            cv2.imshow("Adjusted Image", self.detector.draw_detections(self.image, detections))
            cv2.waitKey(1)

        # Discretize current brightness and contrast for state
        brightness_bin = int((self.current_brightness + 1) / 2 * (self.num_bins - 1))
        contrast_bin = int((self.current_contrast + 1) / 2 * (self.num_bins - 1))
        self.state = (brightness_bin, contrast_bin)

        done = True  # single-step for now
        self.all_rewards.append(reward)
        return self.state, reward, done

    def apply_adjustments(self, img, brightness, contrast):
        beta = brightness * 100
        alpha = 1 + contrast * 0.5
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted

    def plot_rewards(self):
        plt.plot(self.all_rewards)
        plt.title("Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        # Save the plot
        plt.savefig("output/Q/rewards_plot.png")
        plt.show()

        # save rewards to a text file
        with open("output/Q/rewards.txt", "w") as f:
            for reward in self.all_rewards:
                f.write(f"{reward}\n")
        print("Rewards saved to 'rewards.txt' and plot saved as 'rewards_plot.png'.")

if __name__ == "__main__":

    # Q-learning hyperparameters
    num_bins = 10
    num_actions = 9  # 3 x 3 action pairs
    q_table = np.zeros((num_bins, num_bins, num_actions))

    alpha = 0.1     # learning rate
    gamma = 0.9     # discount factor
    epsilon = 1.0   # exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.05
    num_episodes = 10000

    # Setup environment
    model_path = "models/YOLO_eye_detector.pt"
    image_folder = "images/test"
    detector = ObjectDetectorCNN(model_path)
    env = ImagePreprocessingQEnv(detector, image_folder, render=False, num_bins=num_bins)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        action_index = np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(q_table[state[0], state[1]])

        next_state, reward, done = env.step(action_index)

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
        td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - q_table[state[0], state[1], action_index]
        q_table[state[0], state[1], action_index] += alpha * td_error

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {reward:.2f}, Epsilon: {epsilon:.3f}")

    env.plot_rewards()

    print("Q-learning training complete.")

