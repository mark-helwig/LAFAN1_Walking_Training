import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def list_runs():
    """List all available transformer training runs."""
    runs_dir = Path("generated_models/transformer")
    if not runs_dir.exists():
        print("No transformer runs found.")
        return []
    
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    return runs

def view_error_graph(run_path=None, show_latest=False):
    """
    View error graph for a transformer training run.
    
    Args:
        run_path: Path to specific run folder (optional)
        show_latest: If True, show the latest run (default: False, shows all)
    """
    runs = list_runs()
    
    if not runs:
        print("No transformer runs found in generated_models/transformer/")
        return
    
    # Determine which runs to display
    if run_path:
        run_path = Path(run_path)
        if not run_path.exists():
            print(f"Run not found: {run_path}")
            return
        runs_to_display = [run_path]
    elif show_latest:
        runs_to_display = [runs[0]]
    else:
        runs_to_display = runs
    
    # Display available runs
    print(f"Found {len(runs)} transformer training runs:\n")
    for i, run in enumerate(runs):
        marker = " <- LATEST" if i == 0 else ""
        error_file = run / "error.csv"
        if error_file.exists():
            print(f"  {i}: {run.name}{marker}")
        else:
            print(f"  {i}: {run.name} (no error.csv){marker}")
    
    if not run_path and not show_latest:
        print("\nDisplaying all runs...")
        num_plots = len(runs_to_display)
        cols = min(2, num_plots)
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, run in enumerate(runs_to_display):
            error_file = run / "error.csv"
            if error_file.exists():
                df = pd.read_csv(error_file)
                axes[idx].plot(df.index, df['epoch_loss'], linewidth=2)
                axes[idx].set_title(run.name, fontsize=12, fontweight='bold')
                axes[idx].set_xlabel("Epoch")
                axes[idx].set_ylabel("Loss")
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, "error.csv not found", ha='center', va='center')
                axes[idx].set_title(run.name)
        
        # Hide unused subplots
        for idx in range(len(runs_to_display), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
    else:
        # Display single run
        run = runs_to_display[0]
        error_file = run / "error.csv"
        
        if error_file.exists():
            df = pd.read_csv(error_file)
            
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['epoch_loss'], linewidth=2, marker='o', markersize=3)
            plt.title(f"Training Error: {run.name}", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add stats
            min_loss = df['epoch_loss'].min()
            min_epoch = df['epoch_loss'].idxmin()
            final_loss = df['epoch_loss'].iloc[-1]
            
            stats_text = f"Min loss: {min_loss:.6f} (epoch {min_epoch})\nFinal loss: {final_loss:.6f}"
            plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, family='monospace')
        else:
            print(f"error.csv not found in {run}")
            return
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View transformer training error graphs")
    parser.add_argument("--run", type=str, help="Specific run folder to view")
    parser.add_argument("--latest", action="store_true", help="Show only the latest run")
    
    args = parser.parse_args()
    
    view_error_graph(run_path=args.run, show_latest=args.latest)
