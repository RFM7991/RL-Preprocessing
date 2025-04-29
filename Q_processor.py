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
        self.all_differences = []

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
        # Penalize complete failure
        if not adjusted_detections:
            self.all_differences.append(0.0)
            return -1.0

        # Sum confidences for non-eye classes
        original_conf = sum(det['confidence'] for det in original_detections if det['class_name'] != 'Eye')
        adjusted_conf = sum(det['confidence'] for det in adjusted_detections if det['class_name'] != 'Eye')
        delta_conf = adjusted_conf - original_conf
        self.all_differences.append(delta_conf)

        # Optional: log-ratio scaling
        reward = np.log((adjusted_conf + 1e-3) / (original_conf + 1e-3))

        # Optional bonus for new detections
        if len(adjusted_detections) > len(original_detections):
            reward += 0.5  # Additive, not multiplicative

        # Optional clipping to stabilize critic
        reward = np.clip(reward, -1.0, 1.0)

        if self.render:
            print(f"Original Confidence: {original_conf:.2f}, Adjusted Confidence: {adjusted_conf:.2f}")
            print(f"Reward: {reward:.2f}, Brightness: {self.current_beta:.2f}, Contrast: {self.current_alpha:.2f}")
            
            display_img = self.detector.draw_detections(self.image, adjusted_detections)
            cv2.imshow("Adjusted Image", display_img)
            
            key = cv2.waitKey(10)  # Allow more time for GUI to update
            if key == ord('q'):
                self.render = False  # Optional: press 'q' to stop rendering

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

    def plot_rewards(self, save=True, model_type="Q"):
        plt.plot(self.all_rewards)
        plt.title("Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        if save:
            plt.savefig(f"output/{model_type}/rewards_plot.png")
        # plt.show()

        if save: 
            with open("output/Q/rewards.txt", "w") as f:
                for reward in self.all_rewards:
                    f.write(f"{reward}\n")
        # print("Rewards saved to 'rewards.txt' and plot saved as 'rewards_plot.png'.")
        plt.close()
    
    def plot_avg_rewards(self, avg_rewards, save=True, model_type="Q"):
        plt.figure(figsize=(10, 5))
        plt.plot(avg_rewards)
        plt.title("Average Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        if save:
            plt.savefig(f"output/{model_type}/avg_rewards_plot.png")
        # plt.show()

        if save: 
            with open("output/Q/avg_rewards.txt", "w") as f:
                for reward in avg_rewards:
                    f.write(f"{reward}\n")
        plt.close()

    def plot_differences(self, save=True, model_type="Q"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.all_differences)
        plt.title("Differences in Confidence Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Difference in Confidence")
        if save:
            plt.savefig(f"output/{model_type}/differences_plot.png")
        plt.close()
        # plt.show()


    def plot_avg_differences(self, avg_differences, save=True, model_type="Q"):
        plt.figure(figsize=(10, 5))
        plt.plot(avg_differences)
        plt.title("Average Differences in Confidence Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Average Difference in Confidence")
        if save:
            plt.savefig(f"output/{model_type}/avg_differences_plot.png")
        # plt.show()

        plt.close()

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
                past_100_avg_rewards = np.mean(env.all_rewards[-100]) if len(env.all_rewards) >= 100 else np.mean(env.all_rewards)
                print(f"Episode {episode+1}/{num_episodes}, Total Avg.Reward: {np.mean(env.all_rewards):.2f}, 100 eps Avg. Reward: {past_100_avg_rewards:.2}, Epsilon: {epsilon:.3f}")
                env.plot_rewards(save=True, model_type="Q")
            
        rewards.append(env.all_rewards)
        print(f"Experiment {i+1}/{num_experiments}, Avg. Reward: {np.mean(env.all_rewards):.2f}")

    avg_rewards = np.mean(rewards, axis=0)
    print(f"Average Reward over {num_experiments} experiments: {np.mean(avg_rewards):.2f}")
    env.plot_avg_rewards(avg_rewards, save=True, model_type="Q")
    print("Q-learning training complete.")
