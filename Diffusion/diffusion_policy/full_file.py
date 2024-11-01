from dataclasses import dataclass
from typing import Tuple
import os
import sys
import traceback
import torch
import torch.nn.functional as F
import zarr
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import shutil
from diffusers.training_utils import EMAModel


"""
Configurations for data and the model.
"""

@dataclass
class DataConfig:
    """Configurations for dataset"""
    dataset_path: str = None
    pred_horizon: int = 16      # number of future steps to predict
    obs_horizon: int = 2        # number of past observations used to condition the predictions
    action_horizon: int = 8     # number of actions to execute
    # data dimensions
    image_shape: Tuple[int, int, int] = (3, 96, 96)     # size of input images in pixels
    action_dim: int = 2                                  # eg. [vel_x, vel_y]
    state_dim: int = 5                                   # eg. [x,y,z,angle,gripper]


@dataclass
class ModelConfig:
    """Configuration for neural networks"""
    obs_embed_dim: int = 256        # size of observation embedding
    sample_size: int = 16           # length of the generated sequence
    in_channels: int = 2            # action space dimensions
    out_channels: int = 2           # action space dimensions
    layers_per_block: int = 2       # number of conv layers in each UNet block
    block_out_channels: Tuple[int, ...] = (128,)
    norm_num_groups: int = 8
    down_block_types: Tuple[str, ...] = ("DownBlock1D",) * 1
    up_block_types: Tuple[str, ...] = ("UpBlock1D",) * 1

    def __post_init__(self):
        self.total_in_channels = self.in_channels + self.obs_embed_dim // 8


def create_sample_indices(episode_ends: np.ndarray, sequence_length: int,
                         pad_before: int = 0, pad_after: int = 0):
    """Creates valid indices for sampling sequences from episodes."""
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx
            ])
    return np.array(indices)


def sample_sequence(train_data, sequence_length: int,
                   buffer_start_idx: int, buffer_end_idx: int,
                   sample_start_idx: int, sample_end_idx: int):
    """Gets actual data sequence using the indices."""
    result = {}
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    """Computes min and max for normalization."""
    data = data.reshape(-1, data.shape[-1])
    return {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }


def normalize_data(data, stats):
    """Normalize data to [-1, 1] range"""
    return 2.0 * (data - stats['min']) / (stats['max'] - stats['min']) - 1.0


def unnormalize_data(data, stats):
    """Convert from [-1, 1] back to original range"""
    return (data + 1.0) / 2.0 * (stats['max'] - stats['min']) + stats['min']


class SequentialDataset(torch.utils.data.Dataset):
    """Dataset for loading and processing sequential data"""

    def __init__(self, config: DataConfig):
        self.config = config
        dataset_root = zarr.open(config.dataset_path, 'r')
        
        self.train_data = {
            'action': dataset_root['data']['action'][:],
            'state': dataset_root['data']['state'][:],
        }
        if hasattr(dataset_root['data'], 'image'):
            self.train_data['image'] = dataset_root['data']['image'][:]

        self.episode_ends = dataset_root['meta']['episode_ends'][:]
        
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=config.pred_horizon,
            pad_before=config.obs_horizon-1,
            pad_after=config.action_horizon-1
        )

        self.stats = {}
        self.normalized_data = {}
        for key, data in self.train_data.items():
            self.stats[key] = get_data_stats(data)
            self.normalized_data[key] = normalize_data(data, self.stats[key])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, \
        sample_start_idx, sample_end_idx = self.indices[idx]

        sample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.config.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        result = {'action': sample['action']}
        
        if 'state' in sample:
            result['state'] = sample['state'][:self.config.obs_horizon]
        
        if 'image' in sample:
            result['image'] = sample['image'][:self.config.obs_horizon]

        return result


from dataclasses import dataclass
from typing import Tuple
import os
import sys
import traceback
import torch
import torch.nn.functional as F
import zarr
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import shutil
from diffusers.training_utils import EMAModel

@dataclass
class DataConfig:
    """Configurations for dataset"""
    dataset_path: str = None
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    image_shape: Tuple[int, int, int] = (3, 96, 96)
    action_dim: int = 2
    state_dim: int = 5

@dataclass
class ModelConfig:
    """Configuration for neural networks"""
    obs_embed_dim: int = 256
    sample_size: int = 16
    in_channels: int = 2
    out_channels: int = 2
    layers_per_block: int = 2
    block_out_channels: Tuple[int, ...] = (128,)
    norm_num_groups: int = 8
    down_block_types: Tuple[str, ...] = ("DownBlock1D",) * 1
    up_block_types: Tuple[str, ...] = ("UpBlock1D",) * 1

    def __post_init__(self):
        self.total_in_channels = self.in_channels + self.obs_embed_dim // 8

def create_model(data_config: DataConfig, model_config: ModelConfig, device: torch.device):
    """Create model components with corrected architectures"""
    from diffusers import UNet1DModel
    from diffusers.schedulers import DDPMScheduler
    import torch.nn as nn

    # Modified observation encoder to output correct dimensions
    obs_encoder = nn.Sequential(
        nn.Linear(data_config.state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, model_config.obs_embed_dim)
    ).to(device)

    # Modified projection to output correct dimensions
    obs_projection = nn.Sequential(
        nn.Linear(model_config.obs_embed_dim, model_config.obs_embed_dim // 2),
        nn.ReLU(),
        nn.Linear(model_config.obs_embed_dim // 2, model_config.obs_embed_dim // 8)
    ).to(device)

    model = UNet1DModel(
        sample_size=model_config.sample_size,
        in_channels=model_config.total_in_channels,
        out_channels=model_config.out_channels,
        layers_per_block=model_config.layers_per_block,
        block_out_channels=model_config.block_out_channels,
        norm_num_groups=model_config.norm_num_groups,
        down_block_types=model_config.down_block_types,
        up_block_types=model_config.up_block_types,
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )

    return model, obs_encoder, obs_projection, noise_scheduler

def train_diffusion(
    data_config: DataConfig,
    model_config: ModelConfig,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    device: torch.device = None
):
    """Fixed training loop with correct tensor operations"""
    if device is None:
        device = torch.device("cuda")
    print(f"Using device: {device}")

    dataset = SequentialDataset(data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    model, obs_encoder, obs_projection, noise_scheduler = create_model(
        data_config, model_config, device
    )

    model.train()
    obs_encoder.train()
    obs_projection.train()

    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': obs_encoder.parameters()},
        {'params': obs_projection.parameters()}
    ], lr=learning_rate)

    ema = EMAModel(
        parameters=list(model.parameters()) +
                  list(obs_encoder.parameters()) +
                  list(obs_projection.parameters()),
        power=0.75
    )

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}')
        epoch_loss = []

        for batch in dataloader:
            state = batch['state'].to(device, dtype=torch.float32)  # [B, obs_horizon, state_dim]
            actions = batch['action'].to(device, dtype=torch.float32)  # [B, pred_horizon, action_dim]
            batch_size = state.shape[0]

            optimizer.zero_grad()

            try:
                # Process observation sequence
                state_sequence = state.view(batch_size * data_config.obs_horizon, -1)
                obs_embedding = obs_encoder(state_sequence)  # [B*obs_horizon, obs_embed_dim]
                obs_embedding = obs_embedding.view(batch_size, data_config.obs_horizon, -1)
                obs_embedding = obs_embedding.mean(dim=1)  # [B, obs_embed_dim]

                # Project observation embedding
                obs_cond = obs_projection(obs_embedding)  # [B, obs_embed_dim//8]
                
                # Sample noise and timesteps
                noise = torch.randn_like(actions, device=device)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=device
                ).long()

                # Add noise to actions
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                # Reshape tensors to channel format
                noisy_actions = noisy_actions.transpose(1, 2)  # [B, action_dim, pred_horizon]
                noise = noise.transpose(1, 2)  # [B, action_dim, pred_horizon]

                # Expand conditioning to match sequence length
                obs_cond = obs_cond.unsqueeze(-1)  # [B, obs_embed_dim//8, 1]
                obs_cond = obs_cond.expand(-1, -1, noisy_actions.shape[-1])  # [B, obs_embed_dim//8, pred_horizon]

                # Concatenate along channel dimension
                model_input = torch.cat([noisy_actions, obs_cond], dim=1)  # [B, total_in_channels, pred_horizon]

                # Predict noise
                noise_pred = model(model_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

            except RuntimeError as e:
                print(f"\nError in forward pass: {e}")
                print(f"State shape: {state.shape}")
                print(f"Actions shape: {actions.shape}")
                print(f"Obs embedding shape: {obs_embedding.shape}")
                print(f"Obs condition shape: {obs_cond.shape}")
                print(f"Model input shape: {model_input.shape}")
                raise e

            loss.backward()
            optimizer.step()
            ema.step(list(model.parameters()) + 
                    list(obs_encoder.parameters()) +
                    list(obs_projection.parameters()))

            epoch_loss.append(loss.item())
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        progress_bar.close()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"\nEpoch {epoch} average loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                save_dir=save_dir,
                epoch=epoch,
                model=model,
                obs_encoder=obs_encoder,
                obs_projection=obs_projection,
                ema=ema,
                optimizer=optimizer,
                stats=dataset.stats,
                loss=avg_loss,
                filename=f'diffusion_checkpoint_{epoch}.pt'
            )

    save_checkpoint(
        save_dir=save_dir,
        epoch=num_epochs-1,
        model=model,
        obs_encoder=obs_encoder,
        obs_projection=obs_projection,
        ema=ema,
        optimizer=optimizer,
        stats=dataset.stats,
        loss=avg_loss,
        filename='diffusion_final.pt'
    )

    return {
        'model': model,
        'obs_encoder': obs_encoder,
        'obs_projection': obs_projection,
        'ema': ema,
        'noise_scheduler': noise_scheduler,
        'optimizer': optimizer,
        'stats': dataset.stats
    }


def save_checkpoint(save_dir, epoch, model, obs_encoder, obs_projection,
                   ema, optimizer, stats, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': obs_encoder.state_dict(),
        'projection_state_dict': obs_projection.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'loss': loss,
    }, os.path.join(save_dir, filename))


def test_training_setup():
    """Test training initialization"""
    print("\nTesting training setup...")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_config = DataConfig(
        dataset_path=os.path.join(current_dir, 
                                "pusht_cchi_v7_replay.zarr"),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        state_dim=5,
        action_dim=2
    )

    model_config = ModelConfig()
    print("Configurations created successfully")
    return data_config, model_config


def test_mini_training(data_config, model_config):
    """Test a few iterations of training"""
    print("\nTesting mini training...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        save_dir = os.path.join(os.path.dirname(__file__), "saved_model")

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"Removed existing directory: {save_dir}")

        results = train_diffusion(
            data_config=data_config,
            model_config=model_config,
            num_epochs=2,
            batch_size=32,
            device=device,
            save_dir=save_dir
        )

        print("Mini training completed successfully")

        required_keys = ['model', 'obs_encoder', 'obs_projection',
                        'ema', 'noise_scheduler', 'optimizer', 'stats']

        for key in required_keys:
            assert key in results, f"Missing {key} in training results"

        print("All model components returned correctly")

        checkpoint_path = os.path.join(save_dir, "diffusion_final.pt")
        assert os.path.exists(checkpoint_path), "Final checkpoint not saved"

        return results

    except Exception as e:
        print(f"Error in training: {e}")
        traceback.print_exc()
        return None

def test_model_inference(results, model_config):
    """Test if trained model can do inference"""
    print("\nTesting model inference...")

    try:
        model = results['model']
        obs_encoder = results['obs_encoder']
        obs_projection = results['obs_projection']

        device = next(model.parameters()).device
        print(f"Models are on device: {device}")

        # Create a sample batch
        batch_size = 1
        obs_horizon = 2
        state = torch.randn(batch_size, obs_horizon, 5, device=device)  # [B, obs_horizon, state_dim]
        
        with torch.no_grad():
            # Process observation sequence
            state_sequence = state.view(batch_size * obs_horizon, -1)  # [B*obs_horizon, state_dim]
            obs_embedding = obs_encoder(state_sequence)  # [B*obs_horizon, obs_embed_dim]
            obs_embedding = obs_embedding.view(batch_size, obs_horizon, -1)  # [B, obs_horizon, obs_embed_dim]
            obs_embedding = obs_embedding.mean(dim=1)  # [B, obs_embed_dim]
            print(f"Observation embedding shape: {obs_embedding.shape}")

            # Project observation embedding
            obs_cond = obs_projection(obs_embedding)  # [B, obs_embed_dim//8]
            print(f"Observation projection shape: {obs_cond.shape}")

            # Prepare for model input
            pred_horizon = model_config.sample_size
            
            # Reshape conditioning to match the expected input format
            obs_cond = obs_cond.unsqueeze(-1)  # [B, obs_embed_dim//8, 1]
            obs_cond = obs_cond.expand(-1, -1, pred_horizon)  # [B, obs_embed_dim//8, pred_horizon]
            print(f"Expanded conditioning shape: {obs_cond.shape}")

            # Create noisy actions
            noisy_actions = torch.randn(batch_size, model_config.in_channels, pred_horizon, device=device)
            print(f"Noisy actions shape: {noisy_actions.shape}")
            
            # Concatenate along channel dimension
            model_input = torch.cat([noisy_actions, obs_cond], dim=1)  # [B, total_in_channels, pred_horizon]
            print(f"Model input shape: {model_input.shape}")

            # Generate timesteps and run model
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            output = model(model_input, timesteps).sample
            print(f"Model output shape: {output.shape}")

        print("Inference test successful")
        return True

    except Exception as e:
        print(f"Error in inference: {e}")
        traceback.print_exc()
        return False


def create_model(data_config: DataConfig, model_config: ModelConfig, device: torch.device):
    """Create model components"""
    from diffusers import UNet1DModel
    from diffusers.schedulers import DDPMScheduler
    import torch.nn as nn

    # Create observation encoder
    obs_encoder = nn.Sequential(
        nn.Linear(data_config.state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, model_config.obs_embed_dim)
    ).to(device)

    # Create observation projection
    obs_projection = nn.Sequential(
        nn.Linear(model_config.obs_embed_dim, model_config.obs_embed_dim // 2),
        nn.ReLU(),
        nn.Linear(model_config.obs_embed_dim // 2, model_config.obs_embed_dim // 4),
        nn.ReLU(),
        nn.Linear(model_config.obs_embed_dim // 4, model_config.obs_embed_dim // 8)
    ).to(device)

    # Create UNet model
    model = UNet1DModel(
        sample_size=model_config.sample_size,
        in_channels=model_config.total_in_channels,
        out_channels=model_config.out_channels,
        layers_per_block=model_config.layers_per_block,
        block_out_channels=model_config.block_out_channels,
        norm_num_groups=model_config.norm_num_groups,
        down_block_types=model_config.down_block_types,
        up_block_types=model_config.up_block_types,
    ).to(device)

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )

    return model, obs_encoder, obs_projection, noise_scheduler


def main():
    """Main function to run training tests"""
    print("Starting training tests...")

    try:
        # Test setup
        data_config, model_config = test_training_setup()

        # Test training
        results = test_mini_training(data_config, model_config)

        if results is not None:
            # Test inference
            success = test_model_inference(results, model_config)
            if success:
                print("\nAll tests completed successfully!")
            else:
                print("\nInference test failed!")
        else:
            print("\nTraining test failed!")

    except Exception as e:
        print(f"\nError in main: {e}")
        traceback.print_exc()

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()