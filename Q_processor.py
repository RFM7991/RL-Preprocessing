import os
import numpy as np
import cv2
import random
from objectDetectorCNN import ObjectDetectorCNN
from pathlib import Path
import matplotlib.pyplot as plt

class ImagePreprocessingQEnv:
    def __init__(self, detector, image_folder, render=False, num_bins=20):
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

        # Direct value-based brightness and contrast levels
        self.beta_values = np.linspace(-100, 100, num_bins)
        self.alpha_values = np.linspace(0.5, 2.0, num_bins)
        self.action_space = [(b, a) for b in self.beta_values for a in self.alpha_values]
        self.state = (self.num_bins // 2, self.num_bins // 2)  # start from middle

        self.current_beta = 0.0
        self.current_alpha = 1.0
        self.all_rewards = []

    def compute_image_stats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0 * 2 - 1  # Normalize to [-1, 1]
        contrast = (np.std(gray) / 128.0) - 1  # Normalize approx. to [-1, 1]
        return brightness, contrast


    def reset(self):
        self.current_image_index = random.randint(0, len(self.image_paths) - 1)
        self.original_image = cv2.imread(self.image_paths[self.current_image_index])
        if self.original_image is None:
            raise ValueError(f"Could not read image: {self.image_paths[self.current_image_index]}")

        self.image = self.original_image.copy()

        # Measure and set state based on image properties
        brightness, contrast = self.compute_image_stats(self.original_image)
        self.current_beta = brightness * 100
        self.current_alpha = 1 + contrast * 0.5

        beta_bin = np.argmin(np.abs(self.beta_values - self.current_beta))
        alpha_bin = np.argmin(np.abs(self.alpha_values - self.current_alpha))
        self.state = (beta_bin, alpha_bin)

        return self.state

    def get_reward(self, original_detections, adjusted_detections):
        if not adjusted_detections:
            return -1.0
        
        original_confidences = sum(det['confidence'] for det in original_detections)
        adjusted_confidences = sum(det['confidence'] for det in adjusted_detections)
        reward = adjusted_confidences - original_confidences

        if self.render:
            print(f"Original Confidence: {original_confidences:.2f}, Adjusted Confidence: {adjusted_confidences:.2f}")
            print(f"Reward: {reward:.2f}, Brightness: {self.current_beta:.2f}, Contrast: {self.current_alpha:.2f}")
            cv2.imshow(f"Reward: {reward}", self.detector.draw_detections(self.image, adjusted_detections))
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            
        return reward

    def step(self, action_index):
        beta_target, alpha_target = self.action_space[action_index]

        self.current_beta = beta_target
        self.current_alpha = alpha_target

        self.image = self.apply_adjustments(self.original_image, self.current_beta, self.current_alpha)

        original_detections = self.detector.detect_objects(self.detector.preprocess_image_array(self.original_image))
        adjusted_detections = self.detector.detect_objects(self.detector.preprocess_image_array(self.image))

        reward = self.get_reward(original_detections, adjusted_detections)

        beta_bin = np.argmin(np.abs(self.beta_values - self.current_beta))
        alpha_bin = np.argmin(np.abs(self.alpha_values - self.current_alpha))
        self.state = (beta_bin, alpha_bin)

        done = True
        self.all_rewards.append(reward)
        return self.state, reward, done

    def apply_adjustments(self, img, beta, alpha):
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted

    def plot_rewards(self, save=True):
        plt.plot(self.all_rewards)
        plt.title("Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        if save:
            plt.savefig("output/Q/rewards_plot.png")
        # plt.show()

        if save: 
            with open("output/Q/rewards.txt", "w") as f:
                for reward in self.all_rewards:
                    f.write(f"{reward}\n")
        # print("Rewards saved to 'rewards.txt' and plot saved as 'rewards_plot.png'.")

if __name__ == "__main__":
    num_bins = 5
    num_actions = num_bins * num_bins
    q_table = np.zeros((num_bins, num_bins, num_actions))

    alpha = 0.1
    gamma = 1.0
    epsilon = 0.2
    epsilon_decay = 0.9999
    epsilon_min = 0.01
    num_episodes = 10000

    model_path = "models/YOLO_eye_detector.pt"
    image_folder = "images/test"
    detector = ObjectDetectorCNN(model_path)
    env = ImagePreprocessingQEnv(detector, image_folder, render=False, num_bins=num_bins)

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
            print(f"Episode {episode+1}/{num_episodes}, Avg. Reward: {np.mean(env.all_rewards):.2f}, Epsilon: {epsilon:.3f}")
            env.plot_rewards(save=True)
        
    env.plot_rewards()
    print("Q-learning training complete.")
