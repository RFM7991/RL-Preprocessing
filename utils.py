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

# load model and set to eval mode
def load_model(actor_path, critic_path):
    actor = Actor(input_dim=2, output_dim=2)
    critic = Critic(input_dim=2, action_dim=2)
    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))
    actor.eval()
    critic.eval()
   
    return actor, critic

def plot_and_save_results(detections, differences, steps, num_images, num_steps, output_dir="output/DDPG"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(detections, label='Detections')
    plt.xlabel('Episode')
    plt.ylabel('Number of Detections')
    plt.title('Detections Over Episodes')
    plt.legend()
    plt.savefig(f"{output_dir}/plots/eval_detections_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(differences, label='Differences', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Difference in Detections')
    plt.title('Differences in Detections Over Episodes')
    plt.legend()
    plt.savefig(f"{output_dir}/plots/eval_differences_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, label='Steps', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.title('Steps Taken Over Episodes')
    plt.legend()
    plt.savefig(f"{output_dir}/plots/eval_steps_plot.png")

    # Reporting
    results_file = Path(output_dir) / "results/eval_results.txt"
    with open(results_file, "a") as f:
        def write_and_print(s):
            print(s)
            f.write(s + "\n")

        write_and_print(f"\n\nEvaluation Results for {num_images} Episodes and {num_steps} Steps:")

        # Detections stats
        write_and_print(f"Average Detections: {np.mean(detections):.2f}")
        for val in range(4):
            count = sum(1 for d in detections if d == val)
            write_and_print(f"{val} Detections: {count} ({count / len(detections) * 100:.2f}%)")

        # Differences stats
        write_and_print(f"\nAverage Differences: {np.mean(differences):.2f}")
        for val in range(3):
            count = sum(1 for d in differences if d == val)
            write_and_print(f"{val} Differences: {count} ({count / len(differences) * 100:.2f}%)")

        # Steps stats
        write_and_print(f"\nAverage Steps: {np.mean(steps):.2f}")
        count_one = sum(1 for s in steps if s == 1)
        count_max = sum(1 for s in steps if s == num_steps)
        write_and_print(f"One Step: {count_one} ({count_one / len(steps) * 100:.2f}%)")
        write_and_print(f"Max Steps: {count_max} ({count_max / len(steps) * 100:.2f}%)")


def split_dataset(): 
    image_folder = "images/pupils_xor_iris"

        # Get and shuffle all image files
    all_images = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(all_images)

    # Prepare training images
    num_train_images = int(0.8 * len(all_images))
    train_images = all_images[:num_train_images]
    train_images_folder = Path("images/train_shuffle")
    if train_images_folder.exists():
        shutil.rmtree(train_images_folder)
    os.makedirs(train_images_folder)
    for img in train_images:
        src = Path(image_folder) / img
        dst = train_images_folder / img
        shutil.copy(src, dst)
    print(f"Using {len(train_images)} images for training.")

    # Prepare test images
    test_images = all_images[num_train_images:]
    test_images_folder = Path("images/test_shuffle")
    if test_images_folder.exists():
        shutil.rmtree(test_images_folder)
    os.makedirs(test_images_folder)
    for img in test_images:
        src = Path(image_folder) / img
        dst = test_images_folder / img
        shutil.copy(src, dst)
    print(f"Using {len(test_images)} images for testing.")

    return train_images_folder, test_images_folder

def plot_metric_over_episodes(results_dict, metric, save_path, ylabel, title):
    """
    Plots per-episode average of a given metric across step sizes.

    Parameters:
    - results_dict: dict of {step_size: {metric: 2D list}}
    - metric: one of "rewards", "differences", "successful_detections"
    - save_path: where to save the plot
    - ylabel: label for y-axis
    - title: title of the plot
    """
    plt.figure(figsize=(12, 6))

    for step in sorted(results_dict.keys()):
        metric_data = np.array(results_dict[step][metric])  # shape: (num_experiments, num_episodes)
        avg_metric = metric_data.mean(axis=0)               # shape: (num_episodes,)
        plt.plot(avg_metric, label=f"Step {step}")

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_results_dict(results_dict):
    for step in sorted(results_dict.keys()):
        rewards = np.array(results_dict[step]["rewards"]).mean(axis=0)
        diffs = np.array(results_dict[step]["differences"]).mean(axis=0)
        dets = np.array(results_dict[step]["successful_detections"]).mean(axis=0)

        print(f"Step {step} | Avg Reward: {rewards.mean():.2f} | Avg Δ: {diffs.mean():.2f} | Avg Detections: {dets.mean():.2f}")
