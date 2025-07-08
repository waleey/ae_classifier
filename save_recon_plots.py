#Saving the reconstruction to a folder
import os
import matplotlib.pyplot as plt

def save_recon_plots(original_df, reconstruction_df, save_dir, num_samples=None):
    """
    Plots and saves original vs. reconstructed signals from DataFrames.

    Args:
        original_df (pd.DataFrame): DataFrame of shape (N, D), each row is a signal
        reconstruction_df (pd.DataFrame): Same shape as original_df
        save_dir (str): Directory to save plots
        num_samples (int, optional): Number of samples to save (default: all)
    """
    assert original_df.shape == reconstruction_df.shape, "Original and reconstruction must have the same shape"

    os.makedirs(save_dir, exist_ok=True)

    N = len(original_df)
    if num_samples is not None:
        N = min(N, num_samples)

    for i in range(N):
        orig = original_df.iloc[i].values
        recon = reconstruction_df.iloc[i].values

        plt.figure(figsize=(6, 2))
        plt.plot(orig, label='Original', color='blue', linewidth=2)
        plt.plot(recon, label='Reconstruction', color='red', linestyle='--', linewidth=2)
        plt.title(f'Sample {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = os.path.join(save_dir, f'sample_{i:04d}.png')
        plt.savefig(filename, dpi = 50)
        plt.close()
        print(f"Saved plot for sample {i} to {filename}")
