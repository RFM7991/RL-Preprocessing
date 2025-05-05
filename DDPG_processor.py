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
import shutil

# TODO:
# look at reward function
#  evluate the model on a test set of images
# do 20/80 # train/test split
# look for similat results 
# play with steps 
# if needed: add all the YOLO training data, just hold out test set for CNN performance evaluation
# ablate brightness and contrast, see if one is more important than the other
# then explore gamma correction, histogram equalization, CLAHE, etc. 
# try CNN based input for RL 

# save the best model 
# try again for confidence increasing reward for larger data set 
# metric idea - %mulitple detections out of total images / steps 
# should identify any images that never get detected, and then we can look at those images to see if they are problematic
# need to scrub bad images from the dataset, or add more data augmentation to make them more robust
# TODO: add IoU check for pupil in eye.Becomes more relevant with evaluation, but should be incorporated into reward function
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


# load model and set to eval mode
def load_model(actor_path, critic_path):
    actor = Actor(input_dim=2, output_dim=2)
    critic = Critic(input_dim=2, action_dim=2)
    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))
    actor.eval()
    critic.eval()
   
    return actor, critic

def evaluate_model(actor, critic, env, num_episodes, num_steps=1):
    total_rewards = []
    total_detections = []
    total_differences = []
    total_steps = []

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_vector()
        total_reward = 0
        done = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor.to(device)
        critic.to(device)

        for step in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().data.numpy().flatten()
            action += np.random.normal(0, 0.05, size=2)  # small noise for exploration
            action = np.clip(action, -1.0, 1.0)

            beta = action[0] * 100  # scale to [0, 100]
            alpha = action[1] * 0.3 + 1.5  # scale back to [0.5, 1.5]
            env.current_beta = beta
            env.current_alpha = alpha
            env.image = env.apply_adjustments(env.original_image, beta, alpha)

            original_detections = env.detector.detect_objects(env.detector.preprocess_image_array(env.original_image))
            adjusted_detections = env.detector.detect_objects(env.detector.preprocess_image_array(env.image))
            reward = env.get_reward(original_detections, adjusted_detections)

            next_state = env.get_state_vector()
            state = next_state
            total_reward += reward

            print(f"Episode {episode+1}, Step {step+1}, State: {state}, Action: [{beta:.2f}, {alpha:.2f}], Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            if len(adjusted_detections) > len(original_detections):
                break

        
        total_detections.append(len(adjusted_detections) )
        total_differences.append(len(adjusted_detections) - len(original_detections))
        total_steps.append(step + 1)
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    return total_detections, total_differences, total_steps

def run_evaluation(actor_path="models/DDPG_actor.pth", critic_path="models/DDPG_critic.pth", num_steps=1, render=False, model_path="models/YOLO_eye_detector_best.pt", image_folder="images/test"):

    actor, critic = load_model(actor_path, critic_path)

    detector = ObjectDetectorCNN(model_path)
    env = ImagePreprocessingDQNEnv(detector, image_folder, render=render)
    # get count of images in the folder
    num_images = len([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    print(f"Evaluating on {num_images} images.")

    detections, differences, steps = evaluate_model(actor, critic, env, num_episodes=num_images, num_steps=num_steps)

    # plot detections
    plt.figure(figsize=(10, 5))
    plt.plot(detections, label='Detections')
    plt.xlabel('Episode')
    plt.ylabel('Number of Detections')
    plt.title('Detections Over Episodes')
    plt.legend()
    plt.savefig("output/DDPG/eval_detections_plot.png")

    # plot differences
    plt.figure(figsize=(10, 5))
    plt.plot(differences, label='Differences', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Difference in Detections')
    plt.title('Differences in Detections Over Episodes')
    plt.legend()
    plt.savefig("output/DDPG/eval_differences_plot.png")

    # plot steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps, label='Steps', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.title('Steps Taken Over Episodes')
    plt.legend()
    plt.savefig("output/DDPG/eval_steps_plot.png")

    output_dir = Path("output/DDPG")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "eval_results.txt"

    with open(results_file, "a") as f:
        def write_and_print(s):
            print(s)
            f.write(s + "\n")
        
        write_and_print(f"\n\nEvaluation Results for {num_images} Episodes and {num_steps} Steps:")
        # Detections
        avg_detections = np.mean(detections)
        zero_detections = len([d for d in detections if d == 0])
        percent_zero_detections = zero_detections / len(detections) * 100
        one_detection = len([d for d in detections if d == 1])
        percent_one_detection = one_detection / len(detections) * 100
        two_detections = len([d for d in detections if d == 2])
        percent_two_detections = two_detections / len(detections) * 100
        three_detections = len([d for d in detections if d == 3])
        percent_three_detections = three_detections / len(detections) * 100

        write_and_print(f"Average Detections: {avg_detections:.2f}")
        write_and_print(f"Zero Detections: {zero_detections} ({percent_zero_detections:.2f}%)")
        write_and_print(f"One Detection: {one_detection} ({percent_one_detection:.2f}%)")
        write_and_print(f"Two Detections: {two_detections} ({percent_two_detections:.2f}%)")
        write_and_print(f"Three Detections: {three_detections} ({percent_three_detections:.2f}%)")

        # Differences
        avg_differences = np.mean(differences)
        zero_differences = len([d for d in differences if d == 0])
        percent_zero_differences = zero_differences / len(differences) * 100
        one_difference = len([d for d in differences if d == 1])
        percent_one_difference = one_difference / len(differences) * 100
        two_differences = len([d for d in differences if d == 2])
        percent_two_differences = two_differences / len(differences) * 100

        write_and_print(f"\nAverage Differences: {avg_differences:.2f}")
        write_and_print(f"Zero Differences: {zero_differences} ({percent_zero_differences:.2f}%)")
        write_and_print(f"One Difference: {one_difference} ({percent_one_difference:.2f}%)")
        write_and_print(f"Two Differences: {two_differences} ({percent_two_differences:.2f}%)")

        # Steps 
        avg_steps = np.mean(steps)
        one_steps  = len([s for s in steps if s == 1])
        percent_one_steps = one_steps / len(steps) * 100
        max_steps = len([s for s in steps if s == num_steps])
        percent_max_steps = max_steps / len(steps) * 100
        write_and_print(f"\nAverage Steps: {avg_steps:.2f}")
        write_and_print(f"One Step: {one_steps} ({percent_one_steps:.2f}%)")
        write_and_print(f"Max Steps: {max_steps} ({percent_max_steps:.2f}%)")


    print("Evaluation complete.")

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


if __name__ == "__main__":

    # Load the trained model and run evaluation
    # run_evaluation(actor_path="models/DDPG_actor.pth", critic_path="models/DDPG_critic.pth", num_steps=10, render=False)
    # print("Evaluation finished. Check output/DDPG/detections_plot.png for the results.")

    gamma = 1.0
    tau = 0.005
    actor_lr = 1e-3
    critic_lr = 1e-3
    batch_size = 32
    memory_capacity = 100000
    num_episodes = 5000
    num_experiments = 1
    num_steps = 10
    action_noise_std = 0.1
    initial_noise_std = 0.1
    final_noise_std = 0.05
    rewards = []
    differences = []
    successful_detections = []
    # set seed 
    # random.seed(21)
    # np.random.seed(21)
    # torch.manual_seed(21)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(21)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_experiments):
        model_path = "models/YOLO_eye_detector_best.pt"
        train_images_folder, test_images_folder = Path("images/insight/train"), Path("images/insight/test")
        # split_dataset()

        detector = ObjectDetectorCNN(model_path)
        env = ImagePreprocessingDQNEnv(detector, train_images_folder, render=True)

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
                plt.savefig(f"output/DDPG/actions_plot.png")
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


    run_evaluation(actor_path="models/DDPG_actor.pth", critic_path="models/DDPG_critic.pth", num_steps=10, model_path="models/YOLO_eye_detector_best.pt", image_folder="images/test2")
    print("Evaluation finished. Check output/DDPG/detections_plot.png for the results.")

