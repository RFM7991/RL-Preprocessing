import os
import numpy as np
import cv2
import random
from objectDetectorCNN import ObjectDetectorCNN
from pathlib import Path
import matplotlib.pyplot as plt

class ImagePreprocessingQEnv:
    def __init__(self, detector, image_folder, render=False, seed=None):
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.detector = detector
        self.image_folder = Path(__file__).resolve().parent / image_folder
        self.image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        if not self.image_paths:
            raise ValueError(f"No valid images found in {image_folder}")

        self.current_image_index = 0
        self.original_image = None
        self.image = None
        self.image_path = None
        self.render = render

        # Current image transform values (scaled by agent)
        self.current_beta = 0.0
        self.current_alpha = 1.0

        # Logging and tracking variables
        self.all_rewards = []
        self.all_differences = []
        self.successful_detections = []
        self.epoch_counter = 0
        self.p_counter = 0


    def compute_image_stats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0 * 2 - 1  # Normalize to [-1, 1]
        contrast = (np.std(gray) / 128.0) - 1  # Normalize approx. to [-1, 1]
        return brightness, contrast

    def reset(self, shuffle=False):
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_paths):
            self.current_image_index = 0
            self.epoch_counter += 1
            if shuffle:
                random.shuffle(self.image_paths)

        image_path = self.image_paths[self.current_image_index]
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.image = self.original_image.copy()

        brightness, contrast = self.compute_image_stats(self.original_image)
        self.current_beta = brightness * 100
        self.current_alpha = 1 + contrast * 0.5

        self.state = np.array([brightness, contrast])  # continuous state

        self.image_path = image_path
        return self.state


    def get_reward(self, original_detections, adjusted_detections):
        # Sum confidences for non-eye classes
        original_conf = sum(det['confidence'] for det in original_detections if det['class_name']!= 'Eye')
        adjusted_conf = sum(det['confidence'] for det in adjusted_detections if det['class_name']!= 'Eye')

        # Calculate the difference in confidence
        delta_conf = adjusted_conf - original_conf

        # Linear or sigmoid delta scoring
        reward = delta_conf

        if len(original_detections) == 3:
            print(f"Path: {self.image_path}")

        # Optional: confidence-weighted bonus for new detections
        if len(adjusted_detections) > len(original_detections):
            reward += 1
            # Adjust detections based on confidence threshold
            self.successful_detections.append(len(adjusted_detections))

            # Save for analysis
            self.all_differences.append(delta_conf)

        # Optionally apply log-ratio (if you still like it)
        reward = np.log((adjusted_conf + 1e-3) / (original_conf + 1e-3))

        unclipped_reward = reward
        reward = np.clip(reward, -1.0, 1.0)

        if self.render:
            display_img = self.detector.draw_detections(self.image, adjusted_detections)
            cv2.imshow("Adjusted Image", display_img)
            
            key = cv2.waitKey(10)
            if key == ord('q'):
                self.render = False

        return reward

    def step(self, action):
        """
        Executes one step in the environment given a continuous action [beta, alpha].
        Returns: next_state, reward, done
        """
        self.current_beta = action[0] * 100
        self.current_alpha = action[1] * 0.3 + 1.5

        self.image = self.apply_adjustments(self.image, self.current_beta, self.current_alpha)

        original_detections = self.detector.detect_objects(self.detector.preprocess_image_array(self.original_image))
        adjusted_detections = self.detector.detect_objects(self.detector.preprocess_image_array(self.image))
        reward = self.get_reward(original_detections, adjusted_detections)

        brightness, contrast = self.compute_image_stats(self.image)
        next_state = np.array([brightness, contrast])

        self.all_rewards.append(reward)
        self.all_differences.append(reward) 

        done = reward > 0.1
        self.successful_detections.append(len(adjusted_detections))

        return next_state, reward, done, original_detections, adjusted_detections


    def apply_adjustments(self, img, beta, alpha, sharpness=0.0, gamma=1.0):
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if sharpness > 0:
            blurred = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=3)
            adjusted = cv2.addWeighted(adjusted, 1 + sharpness, blurred, -sharpness, 0)

        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
            adjusted = cv2.LUT(adjusted, table)

        return adjusted

    def get_state_vector(self):
        brightness, contrast = self.compute_image_stats(self.image)
        return np.array([brightness, contrast]) 

    def plot_rewards(self, save=True, rewards=[], model_type="Q"):
        plt.figure(figsize=(20, 10))
        plt.plot(rewards)
        plt.title("Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        if save:
            plt.savefig(f"output/{model_type}/plots/rewards_plot.png")
        # plt.show()

        if save: 
            with open(f"output/{model_type}/results/rewards.txt", "w") as f:
                for reward in rewards:
                    f.write(f"{reward}\n")
        plt.close()
    
    def plot_avg_rewards(self, avg_rewards, save=True, model_type="Q"):
        plt.figure(figsize=(20, 10))
        plt.plot(avg_rewards)
        plt.title("Average Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        if save:
            plt.savefig(f"output/{model_type}/plots/avg_rewards_plot.png")
        # plt.show()

        if save: 
            with open(f"output/{model_type}/results/avg_rewards.txt", "w") as f:
                for reward in avg_rewards:
                    f.write(f"{reward}\n")
        plt.close()

    def plot_differences(self, save=True, differences=[], model_type="Q"):
        plt.figure(figsize=(20, 10))
        plt.plot(differences)
        plt.title("Differences in Confidence Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Difference in Confidence")
        if save:
            plt.savefig(f"output/{model_type}/plots/differences_plot.png")
        plt.close()
        # plt.show()


    def plot_avg_differences(self, avg_differences, save=True, model_type="Q"):
        plt.figure(figsize=(20, 10))
        plt.plot(avg_differences)
        plt.title("Average Differences in Confidence Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Average Difference in Confidence")
        if save:
            plt.savefig(f"output/{model_type}/plots/avg_differences_plot.png")

            # Save differences to a text file
            with open(f"output/{model_type}/avg_differences.txt", "w") as f:
                for diff in avg_differences:
                    f.write(f"{diff}\n")

        plt.close()
    
    def plot_successful_detections(self, save=True, detections=[], model_type="Q"):
        plt.figure(figsize=(20, 10))
        plt.plot(detections)
        plt.title("Successful Detections Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Number of Successful Detections")
        if save:
            plt.savefig(f"output/{model_type}/plots/successful_detections_plot.png")
        # plt.show()
        plt.close()

    def plot_avg_successful_detections(self, avg_successful_detections, save=True, model_type="Q"):
        plt.figure(figsize=(20, 10))
        plt.plot(avg_successful_detections)
        plt.title("Average Successful Detections Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Average Number of Successful Detections")
        if save:
            plt.savefig(f"output/{model_type}/plots/avg_successful_detections_plot.png")

            # Save successful detections to a text file
            with open(f"output/{model_type}/results/successful_detections.txt", "w") as f:
                for detection in avg_successful_detections:
                    f.write(f"{detection}\n")
        
        # plt.show()
        plt.close()