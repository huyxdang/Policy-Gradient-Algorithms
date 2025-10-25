# Implementation of a vanilla policy gradient algorithm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym
from torch.optim import Optimizer

# Initialize Model
def create_model(number_of_observations: int, number_of_actions: int):
    """ Create an 1-hidden-layer MLP with ReLu actionvation function

    Args:
        number_of_observations: int, number of observations (input features)
        number_of_actions: int, number of actions (output features)

    Returns: 
        nn.Module: Simple MLP model with 1 hidden layer
    """
    hidden_layer_features = 32
    model = nn.Sequential(
        nn.Linear(in_features = number_of_observations, 
            out_features = hidden_layer_features, bias = True),
        nn.ReLU(),
        nn.Linear(in_features = hidden_layer_features, 
        out_features = number_of_actions, bias = True)
    )
    return model

# Get Policy from Model
def get_policy(model: nn.Module, observation: np.ndarray) -> Categorical:
    " Get policy from a model, given an observation"
    # Convert observation to tensor 
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    # Get logits (probabilities of each action) from model
    logits = model(observation_tensor) # pass observations through model to get logits

    # Applies softmax to get probability distribution of all actions (normalization to [0,1] and sum to 1)
    policy = Categorical(logits=logits)
    return policy # Category type is a probability distribution over all actions

# Sampe Actions from Policy
def sample_actions(policy: Categorical) -> tuple[int, float]: # Return action and its probability
    " Sample actions from a policy, from a specific observation"

    # Sample an action from the policy (unit tensor)
    action = policy.sample() # Note that it does NOT ALWAYS return the action with the highest probability
    # but rather returns an action according to the probability distribution
    # since this allows for exploration (and not just exploitation) over the action space

    # Convert action to an int, as Gym environment expects an int
    action_int = action.item()

    # Log probability of action (for REINFORCE algorithm / loss)
    action_log_prob = policy.log_prob(action)
    
    return action_int, action_log_prob
    
# Calculate Loss 
def calculate_loss(action_log_prob: torch.Tensor, reward: float) -> torch.Tensor:
    """Calculate the 'loss' required to get the policy gradient

    Formula for gradient at
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient

    Note: This isn't the actual loss -- it's just the sum of the log probability
    of each action times the episode return. We calculate this so we can
    back-propagate to get the policy gradient.

    Args:
        epoch_log_probability_actions (torch.Tensor): Log probabilities of the
            actions taken
        epoch_action_rewards (torch.Tensor): Rewards for each of these actions

    Returns:
        float: Pseudo-loss
    """
    return -(epoch_log_probability_actions * epoch_action_rewards).mean()


# Train One Epoch
def train_one_epoch(env: gym.Env, model: nn.Module, optimizer: Optimizer, max_timesteps=5000, episode_timesteps=200) -> float:
    """Train the model for one epoch

    Args:
        env (gym.Env): Gym environment
        model (nn.Module): Model
        optimizer (Optimizer): Optimizer
        max_timesteps (int, optional): Max timesteps per epoch. Note if an
            episode is part-way through, it will still complete before finishing
            the epoch. Defaults to 5000.
        episode_timesteps (int, optional): Timesteps per episode. Defaults to 200.

    Returns:
        float: Average return from the epoch
    """
    epoch_total_timesteps = 0

    # Returns from each episode (to keep track of progress)
    epoch_returns: list[int] = []

    # Action log probabilities and rewards per step (for calculating loss)
    epoch_log_probability_actions = []
    epoch_action_rewards = []

    # Loop through episodes
    while True:

        # Stop if we've done over the total number of timesteps
        if epoch_total_timesteps > max_timesteps:
            break

        # Running total of this episode's rewards
        episode_reward: int = 0

        # Reset the environment and get a fresh observation
        observation = env.reset()

        # Loop through timesteps until the episode is done (or the max is hit)
        for timestep in range(episode_timesteps):
            epoch_total_timesteps += 1

            # Get the policy and act
            policy = get_policy(model, observation)
            action, log_probability_action = get_action(policy)
            observation, reward, done, _ = env.step(action)

            # Increment the episode rewards
            episode_reward += reward

            # Add epoch action log probabilities
            epoch_log_probability_actions.append(log_probability_action)

            # Finish the action loop if this episode is done
            if done == True:
                # Add one reward per timestep
                for _ in range(timestep + 1):
                    epoch_action_rewards.append(episode_reward)

                break

        # Increment the epoch returns
        epoch_returns.append(episode_reward)

    # Calculate the policy gradient, and use it to step the weights & biases
    epoch_loss = calculate_loss(torch.stack(
        epoch_log_probability_actions),
        torch.as_tensor(
        epoch_action_rewards, dtype=torch.float32)
    )

    epoch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return np.mean(epoch_returns)

# Train the Model
def train(epochs=40) -> None:
    """Train a Vanilla Policy Gradient model on CartPole

    Args:
        epochs (int, optional): The number of epochs to run for. Defaults to 50.
    """

    # Create the Gym Environment
    env = gym.make('CartPole-v1')

    # Use random seeds (to make experiments deterministic)
    torch.manual_seed(0)
    np.random.seed(0)

    # Create the MLP model
    number_observation_features = env.observation_space.shape[0]
    number_actions = env.action_space.n
    model = create_model(number_observation_features, number_actions)

    # Create the optimizer
    optimizer = Adam(model.parameters(), 1e-2)

    # Loop for each epoch
    for epoch in range(epochs):
        average_return = train_one_epoch(env, model, optimizer)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))


if __name__ == '__main__':
    train()