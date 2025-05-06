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
    output_dir = Path(output_dir) / str(num_steps)
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Combined plot
    plot_combined_distributions(detections, differences, steps, num_steps, plots_dir)

    # Reporting
    results_file = results_dir / "eval_results.txt"
    with open(results_file, "a") as f:
        def write_and_print(s):
            print(s)
            f.write(s + "\n")

        write_and_print(f"\n\nEvaluation Results for {num_images} Episodes and {num_steps} Steps:")

        write_and_print(f"Average Detections: {np.mean(detections):.2f}")
        for val in range(4):
            count = sum(1 for d in detections if d == val)
            write_and_print(f"{val} Detections: {count} ({count / len(detections) * 100:.2f}%)")

        write_and_print(f"\nAverage Differences: {np.mean(differences):.2f}")
        for val in range(3):
            count = sum(1 for d in differences if d == val)
            write_and_print(f"{val} Differences: {count} ({count / len(differences) * 100:.2f}%)")

        write_and_print(f"\nAverage Steps: {np.mean(steps):.2f}")
        for val in range(1, num_steps + 1):
            count = sum(1 for s in steps if s == val)
            write_and_print(f"{val} Steps: {count} ({count / len(steps) * 100:.2f}%)")


def plot_combined_distributions(detections, differences, steps, num_steps, plots_dir):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    fig.suptitle(f"Evaluation Metrics Distributions for RL Agent Trained for {num_steps} Step Sizes", fontsize=16)

    # Normalize hist heights to percentages
    weights_d = np.ones_like(detections) / len(detections)
    weights_df = np.ones_like(differences) / len(differences)
    weights_s = np.ones_like(steps) / len(steps)

    # Detections
    axs[0].hist(detections, bins=range(0, max(detections) + 2), align='left', color='blue', edgecolor='black', weights=weights_d)
    axs[0].set_title("Detections")
    axs[0].set_xlabel("Number of Detections")
    axs[0].set_ylabel("Percentage")
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Differences
    axs[1].hist(differences, bins=range(min(differences), max(differences) + 2), align='left', color='orange', edgecolor='black', weights=weights_df)
    axs[1].set_title("Confidence Differences")
    axs[1].set_xlabel("Difference in Detections")
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Steps
    axs[2].hist(steps, bins=range(1, num_steps + 2), align='left', color='mediumseagreen', edgecolor='green', weights=weights_s)
    axs[2].set_title("Steps Taken")
    axs[2].set_xlabel("Steps")
    axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    plt.tight_layout()
    plt.savefig(plots_dir / "combined_distribution.png")
    plt.close()



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

