"""
MPC Imitation Learning
Neural network that learns to predict MPC actions from observations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MPCDataset(Dataset):
    """Dataset for MPC imitation learning."""
    
    def __init__(self, observations, actions):
        """
        Args:
            observations: Array of observations (N, obs_dim)
            actions: Array of MPC actions (N, action_dim)
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class MPCImitatorNetwork(nn.Module):
    """Neural network that imitates MPC controller."""
    
    def __init__(self, obs_dim=26, action_dim=6, hidden_dims=[128, 128, 64]):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (joint accelerations)
            hidden_dims: List of hidden layer sizes
        """
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        """
        Forward pass.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            
        Returns:
            actions: Predicted actions (batch_size, action_dim)
        """
        return self.network(obs)


class MPCImitator:
    """Trains and deploys neural network to imitate MPC."""
    
    def __init__(self, obs_dim=32, action_dim=6, hidden_dims=[128, 128, 64]):
        """
        Initialize imitator.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Network architecture
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MPCImitatorNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Normalization statistics
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None
        
    def compute_normalization(self, observations, actions):
        """
        Compute normalization statistics from data.
        
        Args:
            observations: Training observations (N, obs_dim)
            actions: Training actions (N, action_dim)
        """
        self.obs_mean = np.mean(observations, axis=0)
        self.obs_std = np.std(observations, axis=0) + 1e-8
        self.action_mean = np.mean(actions, axis=0)
        self.action_std = np.std(actions, axis=0) + 1e-8
        
    def normalize_obs(self, obs):
        """Normalize observations."""
        return (obs - self.obs_mean) / self.obs_std
    
    def normalize_action(self, action):
        """Normalize actions."""
        return (action - self.action_mean) / self.action_std
    
    def denormalize_action(self, normalized_action):
        """Denormalize actions."""
        return normalized_action * self.action_std + self.action_mean
        
    def train(self, observations, actions, epochs=100, batch_size=64, lr=1e-3, val_split=0.1):
        """
        Train the imitation model.
        
        Args:
            observations: Training observations (N, obs_dim)
            actions: Training actions (N, action_dim)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            val_split: Validation split ratio
            
        Returns:
            history: Training history dict
        """
        # Compute normalization
        self.compute_normalization(observations, actions)
        
        # Normalize data
        obs_norm = self.normalize_obs(observations)
        actions_norm = self.normalize_action(actions)
        
        # Split train/val
        n_val = int(len(obs_norm) * val_split)
        indices = np.random.permutation(len(obs_norm))
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = MPCDataset(obs_norm[train_indices], actions_norm[train_indices])
        val_dataset = MPCDataset(obs_norm[val_indices], actions_norm[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for obs_batch, action_batch in train_loader:
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                
                self.optimizer.zero_grad()
                pred_actions = self.model(obs_batch)
                loss = self.criterion(pred_actions, action_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for obs_batch, action_batch in val_loader:
                    obs_batch = obs_batch.to(self.device)
                    action_batch = action_batch.to(self.device)
                    
                    pred_actions = self.model(obs_batch)
                    loss = self.criterion(pred_actions, action_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return history
    
    def predict(self, observation):
        """
        Predict action from observation.
        
        Args:
            observation: Current observation (obs_dim,)
            
        Returns:
            action: Predicted action (action_dim,)
        """
        self.model.eval()
        with torch.no_grad():
            obs_norm = self.normalize_obs(observation)
            obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
            action_norm = self.model(obs_tensor).cpu().numpy().squeeze()
            action = self.denormalize_action(action_norm)
        return action
    
    def save(self, filepath):
        """Save model and normalization stats."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and normalization stats."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.obs_mean = checkpoint['obs_mean']
        self.obs_std = checkpoint['obs_std']
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        print(f"Model loaded from {filepath}")

