import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from importlib import reload
import sys
import os
sys.path.append('/home/dcollins/repos/')
import dtools_global.vis.pcolormesh_helper as pch
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import loader
import matplotlib as mpl
from scipy.stats import pearsonr
import tqdm
import torch_power
reload(loader)
reload(torch_power)
import nbisht_plotter
reload(nbisht_plotter)
from torch.utils.data import Dataset, DataLoader


def load_processed_MC_data(data_path):
    """
    Load processed MC data ready for model inference.
    
    Args:
        data_path: Path to .npy file from processing script
    
    Returns:
        data_tensor: PyTorch tensor (1, 3, 128, 128) on correct device
    """
    print(f"Loading processed MC data from: {data_path}")
    
    #Load numpy array
    data = np.load(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Data range per channel:")
    for i in range(3):
        print(f"  Channel {i}: [{data[i].min():.3f}, {data[i].max():.3f}]")

    data_tensor = torch.from_numpy(data).float()
    
    data_tensor = data_tensor.unsqueeze(0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_tensor = data_tensor.to(device)
    
    print(f"Tensor shape: {data_tensor.shape}")
    print(f"Device: {device}")
    
    return data_tensor


def load_trained_model(model_path, model_type='net9005'):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to .pth model file
        model_type: 'net9004' (deterministic) or 'net9005' (with uncertainty)
    
    Returns:
        model: Loaded model in eval mode
    """
    print(f"\nLoading {model_type} from: {model_path}")
    
    # Import appropriate network architecture
    if model_type == 'net9004':
        import networks_nbisht.net9004 as net
        has_uncertainty = False
    elif model_type == 'net9005':
        import networks_nbisht.net9005 as net
        has_uncertainty = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model architecture
    model = net.thisnet()
    
    # Load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded successfully")
    print(f"Has uncertainty: {has_uncertainty}")
    
    return model, has_uncertainty


def predict_MC_mach_number(data_tensor, model, has_uncertainty=True):
    """
    Predict Mach number for MC data.
    
    Args:
        data_tensor: Processed data (1, 3, 128, 128)
        model: Trained model
        has_uncertainty: Whether model predicts uncertainty
    
    Returns:
        mean: Predicted Mach number
        std: Uncertainty (standard deviation), or None if deterministic
    """
    print("\nRunning inference...")
    
    with torch.no_grad():
        output = model(data_tensor)
    
    if has_uncertainty:
        # Model outputs (mean, logvar)
        mean, logvar = output
        mean = mean[0][0].cpu().item()
        std = torch.exp(0.5 * logvar[0][0]).cpu().item()
        
        print(f"Prediction: Ms = {mean:.2f} ± {std:.2f}")
        print(f"68% confidence interval: [{mean - std:.2f}, {mean + std:.2f}]")
        print(f"95% confidence interval: [{mean - 2*std:.2f}, {mean + 2*std:.2f}]")
        
        return mean, std
    else:
        # Deterministic model
        mean = output[0][0].cpu().item()
        print(f"Prediction: Ms = {mean:.2f}")
        
        return mean, None


def visualize_MC_with_prediction(data, mean_ms, std_ms=None, 
                                     output_path='./perseus_output/perseus_prediction.png'):
    """
    Create visualization showing Perseus data and Mach number prediction.
    
    Args:
        data: Processed Perseus data (3, 128, 128) numpy array
        mean_ms: Predicted Mach number
        std_ms: Uncertainty (or None)
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Channel 0: Column density
    im0 = axes[0, 0].imshow(data[0], origin='lower', cmap='viridis')
    axes[0, 0].set_title('Column Density Proxy', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('X [pixels]')
    axes[0, 0].set_ylabel('Y [pixels]')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Channel 1: Velocity-weighted
    im1 = axes[0, 1].imshow(data[1], origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('Velocity-Weighted Density', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('X [pixels]')
    axes[0, 1].set_ylabel('Y [pixels]')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Channel 2: Velocity dispersion
    im2 = axes[1, 0].imshow(data[2], origin='lower', cmap='plasma')
    axes[1, 0].set_title('Velocity Dispersion', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('X [pixels]')
    axes[1, 0].set_ylabel('Y [pixels]')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Prediction summary
    axes[1, 1].axis('off')
    
    if std_ms is not None:
        prediction_text = f"""
        MOLECULAR CLOUD
        Sonic Mach Number Prediction
        
        ═══════════════════════════════
        
        Predicted Ms: {mean_ms:.2f} ± {std_ms:.2f}
        
        Confidence Intervals:
        ───────────────────────────────
        68% CI: [{mean_ms - std_ms:.2f}, {mean_ms + std_ms:.2f}]
        95% CI: [{mean_ms - 2*std_ms:.2f}, {mean_ms + 2*std_ms:.2f}]
        
        ═══════════════════════════════
        
        Physical Interpretation:
        ───────────────────────────────
        """
        
        # Add physical interpretation based on Mach number
        if mean_ms < 1:
            regime = "SUBSONIC"
            description = "Turbulence is dominated by\nthermal pressure. Flow is\nsmooth with weak shocks."
        elif mean_ms < 3:
            regime = "TRANSONIC"
            description = "Turbulence is marginally\nsupersonic. Moderate shock\nformation expected."
        elif mean_ms < 5:
            regime = "SUPERSONIC"
            description = "Strong supersonic turbulence.\nSignificant shock networks\nand density contrast."
        else:
            regime = "HIGHLY SUPERSONIC"
            description = "Extreme turbulence with\ncomplex shock structures.\nHigh star formation potential."
        
        prediction_text += f"Regime: {regime}\n\n{description}"
        
    else:
        prediction_text = f"""
        MOLECULAR CLOUD
        Sonic Mach Number Prediction
        
        ═══════════════════════════════
        
        Predicted Ms: {mean_ms:.2f}
        
        (Deterministic model - no uncertainty)
        """
    
    axes[1, 1].text(0.1, 0.5, prediction_text,
                   transform=axes[1, 1].transAxes,
                   fontsize=11,
                   verticalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    fig.suptitle('Molecular Cloud: ML-Based Mach Number Inference',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    plt.close()


def compare_models(data_tensor, models_dict):
    """
    Run inference with multiple models and compare predictions.
    
    Args:
        data_tensor: Processed PERSEUS data
        models_dict: Dict of {model_name: (model, has_uncertainty)}
    
    Returns:
        results: Dict of {model_name: (mean, std)}
    """
    print("COMPARING MULTIPLE MODELS")
    
    results = {}
    
    for model_name, (model, has_uncertainty) in models_dict.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        with torch.no_grad():
            output = model(data_tensor)
        
        if has_uncertainty:
            mean, logvar = output
            mean = mean[0][0].cpu().item()
            std = torch.exp(0.5 * logvar[0][0]).cpu().item()
            results[model_name] = (mean, std)
            print(f"Ms = {mean:.2f} ± {std:.2f}")
        else:
            mean = output[0][0].cpu().item()
            results[model_name] = (mean, None)
            print(f"Ms = {mean:.2f}")
    
    # Summary comparison
    print("SUMMARY")
    
    means = [r[0] for r in results.values()]
    print(f"Mean prediction: {np.mean(means):.2f}")
    print(f"Std across models: {np.std(means):.2f}")
    print(f"Range: [{min(means):.2f}, {max(means):.2f}]")
    
    return results



def main(MC_PATH, MC_MODEL_INPUT, obs_name = 'Perseus'):
    print("Load Processed MC Data")
    data_tensor = load_processed_MC_data(MC_MODEL_INPUT)
    compare_models = 0
    model_name = '9005'
    if compare_models:
        #Load multiple models for comparison
        print("Load Multiple Models")
        
        models_dict = {}
        
        # Try to load different model versions
        model_paths = {
            'Net9004 (Hybrid CNN-ViT)': './models/test9004.pth',
            'Net9005 (With Uncertainty)': './models/test9005.pth',
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                model_type = 'net9005' if '9005' in path else 'net9004'
                try:
                    model, has_unc = load_trained_model(path, model_type)
                    models_dict[name] = (model, has_unc)
                except Exception as e:
                    print(f"Could not load {name}: {e}")
        
        #Run comparison
        print("STEP 3: Model Comparison")
        results = compare_models(data_tensor, models_dict)
        
        #Use best model (with uncertainty if available)
        best_model_name = list(models_dict.keys())[0]
        model, has_uncertainty = models_dict[best_model_name]
        mean_ms, std_ms = results[best_model_name]
        
    else:
        #Load single model
        print("Load Model")
        model, has_uncertainty = load_trained_model(model_path = f'./models/test{model_name}.pth', model_type=f'net{model_name}')
        
        print("Predict Mach Number")
        mean_ms, std_ms = predict_MC_mach_number(data_tensor, model, has_uncertainty)
    

    print("Visualize Results")
    data_np = np.load(MC_MODEL_INPUT)
    visualize_MC_with_prediction(data_np, mean_ms, std_ms,
                                     output_path=f'{MC_PATH}/{obs_name}_prediction.png')
    
    results_file = f'{MC_PATH}/{obs_name}_results.txt'
    with open(results_file, 'w') as f:
        f.write("MOLECULAR CLOUD - MACH NUMBER PREDICTION\n")
        
        if std_ms is not None:
            f.write(f"Predicted Sonic Mach Number: {mean_ms:.2f} ± {std_ms:.2f}\n\n")
            f.write(f"68% Confidence Interval: [{mean_ms - std_ms:.2f}, {mean_ms + std_ms:.2f}]\n")
            f.write(f"95% Confidence Interval: [{mean_ms - 2*std_ms:.2f}, {mean_ms + 2*std_ms:.2f}]\n")
        else:
            f.write(f"Predicted Sonic Mach Number: {mean_ms:.2f}\n")
        
        f.write(f"\nModel: net{model_name}\n")
        f.write(f"Model Path: ./models/test{model_name}.pth\n")
        f.write(f"Data Path: {PERSEUS_13CO_MODEL_INPUT}\n")
    
    print(f"\nResults saved: {results_file}")
    
    print("INFERENCE COMPLETE")
    print(f"\nPredicted Mach number for Perseus: Ms = {mean_ms:.2f}" + 
          (f" ± {std_ms:.2f}" if std_ms is not None else ""))


if __name__ == "__main__":
    PERSEUS_13CO_PATH = '../data/perseus_mc/'
    PERSEUS_13CO_MOMENT_PATH = PERSEUS_13CO_PATH + 'PerA_13coFCRAO_F_map.fits.gz'
    PERSEUS_13CO_PPV_PATH = PERSEUS_13CO_PATH + 'PerA_13coFCRAO_F_xyv.fits.gz'
    PERSEUS_13CO_MODEL_INPUT = PERSEUS_13CO_PATH + 'perseus_model_input.npy'

    TAURUS_13CO_PATH = '../data/taurus_mc/'
    TAURUS_13CO_MOMENT_PATH = TAURUS_13CO_PATH + 'DHT21_Taurus_mom.fits'
    TAURUS_13CO_PPV_PATH = TAURUS_13CO_PATH + 'DHT21_Taurus_interp.fits'
    TAURUS_13CO_MODEL_INPUT = TAURUS_13CO_PATH + 'taurus_model_input.npy'

    main(TAURUS_13CO_PATH, TAURUS_13CO_MODEL_INPUT, obs_name='taurus')