import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
import tqdm
import os


def plot_scalar_with_uncertainty(net_name, train_loader, val_loader, tst_loader, model):
    """
    Enhanced plotter for models with uncertainty estimation.
    
    Shows:
    1. Predictions with error bars
    2. Calibration metrics (% within 1σ, 2σ, 3σ)
    3. Uncertainty vs Mach number
    4. Color-coded by calibration quality
    """
    
    subs = ['test']
    device = 'cuda'
    
    for ns, subset in enumerate(subs):
        this_set = {'train': train_loader, 'valid': val_loader, 'test': tst_loader}[subset]
        
        ms_net = []
        ms_target = []
        ms_std = []  # Uncertainty (standard deviation)
        
        print(f"\nEvaluating {subset} set with uncertainty...")
        
        with torch.no_grad():
            for xb, yb in tqdm.tqdm(this_set):
                ms_true = yb[0][0].cpu().item()
                
                # Model returns (mean, logvar)
                model_out = model(xb)
                
                if isinstance(model_out, tuple):
                    mean, logvar = model_out
                    mean = mean[0][0].cpu().item()
                    # Convert log variance to standard deviation
                    std = torch.exp(0.5 * logvar[0][0]).cpu().item()
                else:
                    # Fallback for models without uncertainty
                    mean = model_out[0][0].cpu().item()
                    std = 0.0
                
                ms_target.append(ms_true)
                ms_net.append(mean)
                ms_std.append(std)
        
        ms_target = np.array(ms_target)
        ms_net = np.array(ms_net)
        ms_std = np.array(ms_std)
        
        # Calculate metrics
        pearson_r = pearsonr(ms_target, ms_net)[0]
        mae = np.abs(ms_target - ms_net).mean()
        rmse = np.sqrt(((ms_target - ms_net)**2).mean())
        
        # Calibration: % of predictions within k*sigma
        errors = np.abs(ms_target - ms_net)
        within_1sigma = (errors <= 1.0 * ms_std).mean() * 100
        within_2sigma = (errors <= 2.0 * ms_std).mean() * 100
        within_3sigma = (errors <= 3.0 * ms_std).mean() * 100
        
        print(f"\n{subset.capitalize()} Set Metrics:")
        print(f"  Pearson R: {pearson_r:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"\nCalibration (ideal: 68%, 95%, 99.7%):")
        print(f"  Within 1σ: {within_1sigma:.1f}%")
        print(f"  Within 2σ: {within_2sigma:.1f}%")
        print(f"  Within 3σ: {within_3sigma:.1f}%")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ============================================
        # Plot 1: Main scatter with error bars
        # ============================================
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Color-code by calibration quality
        calibration_quality = errors / (ms_std + 1e-8)  # Normalized error
        
        scatter = ax1.scatter(ms_target, ms_net, c=calibration_quality, 
                            cmap='RdYlGn_r', vmin=0, vmax=3, 
                            s=20, alpha=0.6, edgecolors='k', linewidths=0.5)
        
        # Add error bars (subsample for clarity)
        n_samples = len(ms_target)
        step = max(1, n_samples // 200)  # Show ~200 error bars
        idx = np.arange(0, n_samples, step)
        ax1.errorbar(ms_target[idx], ms_net[idx], yerr=2*ms_std[idx], 
                    fmt='none', ecolor='gray', alpha=0.3, linewidth=1, 
                    label='±2σ uncertainty')
        
        # Perfect prediction line
        maxs = max(ms_target.max(), ms_net.max())
        ax1.plot([0, maxs], [0, maxs], 'k--', linewidth=2, alpha=0.5, label='Perfect')
        
        ax1.set_xlabel('True Ms', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted Ms', fontsize=14, fontweight='bold')
        ax1.set_title(f'{subset.capitalize()} Set - Pearson R = {pearson_r:.4f}', 
                     fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Error / σ (lower is better)', fontsize=11)
        
        # ============================================
        # Plot 2: Uncertainty vs Mach Number
        # ============================================
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Bin by Mach number and show average uncertainty
        bins = np.linspace(0, ms_target.max(), 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_std_mean = []
        bin_std_err = []
        
        for i in range(len(bins) - 1):
            mask = (ms_target >= bins[i]) & (ms_target < bins[i+1])
            if mask.sum() > 0:
                bin_std_mean.append(ms_std[mask].mean())
                bin_std_err.append(ms_std[mask].std() / np.sqrt(mask.sum()))
            else:
                bin_std_mean.append(0)
                bin_std_err.append(0)
        
        ax2.errorbar(bin_centers, bin_std_mean, yerr=bin_std_err, 
                    fmt='o-', linewidth=2, markersize=6, capsize=4)
        ax2.set_xlabel('True Ms', fontsize=11)
        ax2.set_ylabel('Avg Uncertainty (σ)', fontsize=11)
        ax2.set_title('Uncertainty vs Mach', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ============================================
        # Plot 3: Calibration Plot
        # ============================================
        ax3 = fig.add_subplot(gs[1, 2])
        
        # Expected vs observed coverage at different sigma levels
        sigma_levels = np.linspace(0, 3, 30)
        observed_coverage = []
        for sigma in sigma_levels:
            coverage = (errors <= sigma * ms_std).mean() * 100
            observed_coverage.append(coverage)
        
        # Ideal Gaussian calibration
        from scipy.stats import norm
        ideal_coverage = [norm.cdf(sigma) * 100 for sigma in sigma_levels]
        
        ax3.plot(sigma_levels, ideal_coverage, 'k--', linewidth=2, label='Ideal (Gaussian)')
        ax3.plot(sigma_levels, observed_coverage, 'b-', linewidth=2, label='Observed')
        ax3.fill_between(sigma_levels, ideal_coverage, observed_coverage, 
                        alpha=0.3, color='blue')
        
        ax3.set_xlabel('Confidence Level (σ)', fontsize=11)
        ax3.set_ylabel('Coverage (%)', fontsize=11)
        ax3.set_title('Calibration Curve', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 3])
        ax3.set_ylim([0, 100])
        
        # ============================================
        # Plot 4: Residuals with uncertainty bands
        # ============================================
        ax4 = fig.add_subplot(gs[2, :])
        
        # Sort by Mach number for better visualization
        sort_idx = np.argsort(ms_target)
        ms_sorted = ms_target[sort_idx]
        residuals_sorted = (ms_net - ms_target)[sort_idx]
        std_sorted = ms_std[sort_idx]
        
        # Subsample for clarity
        step = max(1, len(ms_sorted) // 500)
        idx = np.arange(0, len(ms_sorted), step)
        
        ax4.scatter(ms_sorted[idx], residuals_sorted[idx], c='blue', s=10, alpha=0.5)
        ax4.fill_between(ms_sorted[idx], -2*std_sorted[idx], 2*std_sorted[idx], 
                        alpha=0.3, color='red', label='±2σ band')
        ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        ax4.set_xlabel('True Ms', fontsize=12)
        ax4.set_ylabel('Residual (Pred - True)', fontsize=12)
        ax4.set_title('Residuals with Uncertainty Bands', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Save figure
        fig.suptitle(f'Uncertainty Analysis - {net_name}', fontsize=18, fontweight='bold', y=0.995)
        oname = f"{os.environ['HOME']}/plots/{net_name}_uncertainty_{subset}.png"
        fig.savefig(oname, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {oname}")
        plt.close(fig)
        
        # ============================================
        # Binned Statistics Table
        # ============================================
        print(f"\nBinned Performance:")
        bins = [0, 3, 6, 9, 12, 20]
        for i in range(len(bins)-1):
            mask = (ms_target >= bins[i]) & (ms_target < bins[i+1])
            if mask.sum() > 0:
                bin_pearson = pearsonr(ms_target[mask], ms_net[mask])[0]
                bin_mae = np.abs(ms_target[mask] - ms_net[mask]).mean()
                bin_std_avg = ms_std[mask].mean()
                bin_calib = (np.abs(ms_target[mask] - ms_net[mask]) <= ms_std[mask]).mean() * 100
                bin_count = mask.sum()
                
                print(f"Ms [{bins[i]:2d}-{bins[i+1]:2d}): N={bin_count:4d}, R={bin_pearson:.4f}, "
                      f"MAE={bin_mae:.4f}, σ_avg={bin_std_avg:.4f}, within_1σ={bin_calib:.1f}%")


def plot_scalar_onlyms(net_name, train_loader, val_loader, tst_loader, model):
    """
    Backward-compatible plotter for models WITHOUT uncertainty.
    Auto-detects if model has uncertainty and calls appropriate plotter.
    """
    
    # Test if model outputs uncertainty
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    # Get one sample to check output format
    for xb, yb in tst_loader:
        with torch.no_grad():
            model_out = model(xb)
        break
    
    if isinstance(model_out, tuple) and len(model_out) == 2:
        # Model has uncertainty
        print("Detected uncertainty model - using enhanced plotter")
        plot_scalar_with_uncertainty(net_name, train_loader, val_loader, tst_loader, model)
    else:
        # Original plotter for models without uncertainty
        print("Detected standard model - using basic plotter")
        _plot_scalar_basic(net_name, train_loader, val_loader, tst_loader, model)


def _plot_scalar_basic(net_name, train_loader, val_loader, tst_loader, model):
    """Original plotter for models without uncertainty (backward compatibility)"""
    import dtools_global.vis.pcolormesh_helper as pch
    
    subs = ['test']
    device = 'cuda'
    ms_net = []
    ms_target = []
    
    for ns, subset in enumerate(subs):
        this_set = {'train': train_loader, 'valid': val_loader, 'test': tst_loader}[subset]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        maxs = 0
        with torch.no_grad():
            for xb, yb in tqdm.tqdm(this_set):
                ms = yb[0][0].cpu()
                moo = model(xb)
                
                if isinstance(moo, tuple):
                    ms_moo = moo[0][0][0].cpu()  # Extract mean if uncertainty model
                else:
                    ms_moo = moo[-1][0].cpu()
                
                ms_net.append(ms_moo.item())
                ms_target.append(ms.item())
                maxs = max([maxs, ms.item()])
        
        if len(ms_target) < 100:
            for a, b in zip(ms_target, ms_net):
                ax.scatter(a, b)
        else:
            pch.simple_phase(ms_target, ms_net, ax=ax)
        
        pearson_r = pearsonr(ms_target, ms_net)[0]
        ax.set_title(f'{subset.capitalize()} Set - Pearson R = {pearson_r:.4f}', fontsize=14)
        ax.set_xlabel('True Ms', fontsize=12)
        ax.set_ylabel('Predicted Ms', fontsize=12)
        ax.plot([0, maxs], [0, maxs], 'k--', linewidth=2)
        ax.grid(True, alpha=0.3)
        
        fig.savefig(f"{os.environ['HOME']}/plots/{net_name}_scalars_{subset}.png")
        plt.close(fig)
    
    # Binned statistics
    ms_target_np = np.array(ms_target)
    ms_net_np = np.array(ms_net)
    
    bins = [0, 3, 6, 9, 12, 20]
    for i in range(len(bins)-1):
        mask = (ms_target_np >= bins[i]) & (ms_target_np < bins[i+1])
        if mask.sum() > 0:
            bin_pearson = pearsonr(ms_target_np[mask], ms_net_np[mask])[0]
            bin_mae = np.abs(ms_target_np[mask] - ms_net_np[mask]).mean()
            bin_count = mask.sum()
            print(f"Ms [{bins[i]}-{bins[i+1]}): N={bin_count:4d}, R={bin_pearson:.4f}, MAE={bin_mae:.4f}")