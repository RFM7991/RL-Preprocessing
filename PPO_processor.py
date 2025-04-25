import os
import gym
from gym import spaces
import numpy as np
import cv2
import random
from stable_baselines3 import PPO
from objectDetectorCNN import ObjectDetectorCNN
from pathlib import Path
import matplotlib.pyplot as plt


class ImagePreprocessingEnv(gym.Env):
    def __init__(self, detector, image_folder, render=False):
        super(ImagePreprocessingEnv, self).__init__()
        self.detector = detector
        self.image_folder = Path(__file__).resolve().parent / image_folder
        self.image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        if not self.image_paths:
            raise ValueError(f"No valid images found in {image_folder}")
        
        self.current_image_index = 0
        self.original_image = None
        self.image = None
        self.render = render

        # Define action space: brightness [-1, 1], contrast [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Use a sample image to set observation space
        sample_image = cv2.imread(self.image_paths[0])
        self.observation_space = spaces.Box(low=0, high=255, shape=sample_image.shape, dtype=np.uint8)

        self.all_rewards = []

    def reset(self):
        self.current_image_index = random.randint(0, len(self.image_paths) - 1)
        self.original_image = cv2.imread(self.image_paths[self.current_image_index])
        if self.original_image is None:
            raise ValueError(f"Could not read image: {self.image_paths[self.current_image_index]}")
        
        self.image = self.original_image.copy()
        return self.image

    def step(self, action):
        brightness_adj, contrast_adj = action
        self.image = self.apply_adjustments(self.original_image, brightness_adj, contrast_adj)

        img_tensor = self.detector.preprocess_image_array(self.image)
        detections = self.detector.detect_objects(img_tensor)

        # Reward: sum of confidences
        if detections:
                reward = sum(det['confidence'] for det in detections) 

                # add penalty for extra detections
                if len(detections) > 3:
                    reward += (3 - len(detections))
        else:
            reward = -1.0

        if self.render:
            print(f"Reward: {reward:.4}, Detections: {len(detections)}, Brightness: {brightness_adj:.2f}, Contrast: {contrast_adj:.2f}")

        # draw detections on the image
        if self.render:
            cv2.imshow("Original Image", self.original_image)
            cv2.imshow("Detections", self.detector.draw_detections(self.image, detections))
            cv2.waitKey(1)
        
        done = True  # Single-step episode or make multi-step
        obs = self.image
        info = {"detections": detections, "image_path": self.image_paths[self.current_image_index]}

        self.all_rewards.append(reward)
        return obs, reward, done, info

    def apply_adjustments(self, img, brightness, contrast):
        beta = brightness * 100   # Brightness adjustment [-100, 100]
        alpha = 1 + contrast * 0.5  # Contrast scaling [0.5, 1.5]

        # Apply brightness and contrast first
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # # Apply gamma correction
        # gamma = np.clip(1 + gamma, 0.1, 3.0)  # Map [-1, 1] to [0.1, 3.0]
        # inv_gamma = 1.0 / gamma
        # table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        # adjusted = cv2.LUT(adjusted, table)

        return adjusted


    def plot_rewards(self):
        plt.plot(self.all_rewards)
        plt.title("Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        # Save the plot
        plt.savefig("output/PPO/rewards_plot.png")
        plt.show()

        # save rewards to a text file
        with open("output/PPO/rewards.txt", "w") as f:
            for reward in self.all_rewards:
                f.write(f"{reward}\n")
        print("Rewards saved to 'rewards.txt' and plot saved as 'rewards_plot.png'.")

if __name__ == "__main__":

    model_path = "models/YOLO_eye_detector.pt"
    image_folder = "images/test"

    detector = ObjectDetectorCNN(model_path)
    env = ImagePreprocessingEnv(detector, image_folder, render=False)

    # check for GPU availability
    device = detector.device
    print(f"Using device: {device}")

    n_steps = 64  # Number of steps per rollout
    batch_size = 16  # Size of mini-batches
    total_timesteps = 10000  # Total timesteps for training
    n_epochs = 10  # Number of epochs per update
    learning_rate = 3e-4  # Learning rate

    model = PPO("CnnPolicy", env, verbose=1, device=device, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Evaluate the model
    env.plot_rewards()

    # save the model
    model.save("models/image_preprocessing_model")
    print("Model training complete. Model saved as 'image_preprocessing_model'.")
