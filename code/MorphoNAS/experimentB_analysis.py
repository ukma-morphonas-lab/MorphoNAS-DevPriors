import os
import json
import glob
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def process_experiment_results(experiment_path, experiment_name, show_generations=True):
    # Construct the pattern to match result directories
    pattern = os.path.join(experiment_path, "results", f"{experiment_name}_R*")
    result_dirs = glob.glob(pattern)
    
    if not result_dirs:
        print(f"No result directories found matching pattern: {pattern}")
        return
    
    # List to store data from each run
    runs_data = []
    
    for result_dir in result_dirs:
        stats_file = os.path.join(result_dir, "stats.json")
        if not os.path.exists(stats_file):
            print(f"Stats file not found in {result_dir}")
            continue
            
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Get the last generation data
            last_gen = stats['generations_data'][-1]
            
            # Extract run number from directory name
            run_number = int(os.path.basename(result_dir).split('_R')[-1])
            
            # Create data entry for this run
            run_data = {
                'run': run_number,
                'generations': stats['generations'],
                'best_fitness': stats['best_fitness'],
                'evaluations': stats['evaluations'],
                'elapsed_time': stats['elapsed_time'],
                'neurons': last_gen['neurons'],
                'inputs': last_gen['inputs'],
                'connections': last_gen['connections'],
                'distinct_fitnesses': last_gen['distinct_fitness_scores'],
                'distinct_genomes': last_gen['distinct_genomes'],
                'avg_fitness': last_gen['avg_fitness']
            }
            
            runs_data.append(run_data)
            
        except Exception as e:
            print(f"Error processing {stats_file}: {str(e)}")
    
    if not runs_data:
        print("No valid data found in any result directory")
        return
    
    # Create DataFrame and sort by run number
    df = pd.DataFrame(runs_data)
    df = df.sort_values('run')
    
    # Create analysis directory
    analysis_dir = os.path.join(experiment_path, "results", "analysis", experiment_name)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate output filename
    output_file = os.path.join(analysis_dir, f"{experiment_name}_analysis.csv")
    df.to_csv(output_file, index=False)
    print(f"Analysis saved to {output_file}")
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 16,  # Base font size
        'axes.labelsize': 20,  # Axis labels
        'axes.titlesize': 24,  # Title
        'xtick.labelsize': 16,  # X-axis tick labels
        'ytick.labelsize': 16,  # Y-axis tick labels
    })
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Set both axes to logarithmic scale
    plt.yscale('log')
    plt.xscale('log')
    
    if show_generations:
        # Calculate percentages for legend
        zero_gen_count = (df['generations'] == 0).sum()
        total_count = len(df)
        zero_gen_percent = (zero_gen_count / total_count) * 100
        non_zero_percent = 100 - zero_gen_percent
        
        # Plot points with 0 generations in red
        zero_gen_mask = df['generations'] == 0
        plt.scatter(df[zero_gen_mask]['neurons'], df[zero_gen_mask]['connections'], 
                   alpha=0.6, s=120, color='red', marker='o', 
                   label=f'0 generations ({zero_gen_percent:.0f}%)')
        
        # Plot other points in black with bold cross marker
        plt.scatter(df[~zero_gen_mask]['neurons'], df[~zero_gen_mask]['connections'], 
                   alpha=0.6, s=120, marker='x', color='black', linewidth=2, 
                   label=f'>0 generations ({non_zero_percent:.0f}%)')
        
        # Add legend at bottom right
        plt.legend(loc='lower right')
    else:
        # Plot all points in light blue
        plt.scatter(df['neurons'], df['connections'], 
                   alpha=0.6, s=120, marker='o')
    
    plt.xlabel('Number of Neurons (log scale)')
    plt.ylabel('Number of Connections (log scale)')
    plt.title(f'RNN Size Distribution - {experiment_name}')
    
    # Get the current axis limits
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    # Generate custom tick positions for Y-axis
    min_power_y = np.floor(np.log10(ymin))
    max_power_y = np.ceil(np.log10(ymax))
    
    # Create tick positions at each power of 10 and halfway between for Y-axis
    tick_positions_y = []
    tick_labels_y = []
    for power in range(int(min_power_y), int(max_power_y) + 1):
        tick_positions_y.append(10**power)
        tick_labels_y.append(f'{int(10**power)}')
        if power < max_power_y:  # Don't add intermediate tick after the last power
            tick_positions_y.append(3 * 10**power)
            tick_labels_y.append(f'{int(3 * 10**power)}')
    
    # Generate custom tick positions for X-axis
    min_power_x = np.floor(np.log10(xmin))
    max_power_x = np.ceil(np.log10(xmax))
    
    # Create tick positions at each power of 10 for X-axis
    tick_positions_x = []
    tick_labels_x = []
    for power in range(int(min_power_x), int(max_power_x) + 1):
        tick_positions_x.append(10**power)
        tick_labels_x.append(f'{int(10**power)}')
    
    # Set custom ticks
    plt.yticks(tick_positions_y, tick_labels_y)
    plt.xticks(tick_positions_x, tick_labels_x)
    
    # Add major grid lines
    plt.grid(True, linestyle='--', alpha=0.7, which='major')
    
    # Add minor grid lines
    plt.grid(True, linestyle=':', alpha=0.7, which='minor')
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    
    # Save the first plot
    plot_file = os.path.join(analysis_dir, f"{experiment_name}_network_size.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_file}")
    
    # Create second plot: Connections vs Generations
    plt.figure(figsize=(10, 8))
    
    # Set both axes to logarithmic scale
    plt.yscale('log')
    plt.xscale('log')
    
    # Create a copy of generations data, replacing 0 with 0.1 for plotting
    generations_for_plot = df['generations'].copy()
    generations_for_plot[generations_for_plot == 0] = 0.1
    
    # Plot all points in light blue
    plt.scatter(generations_for_plot, df['connections'], 
               alpha=0.6, s=120, marker='o')
    
    plt.xlabel('Number of Generations (log scale)')
    plt.ylabel('Number of Connections (log scale)')
    plt.title(f'Network Size vs Generations - {experiment_name}')
    
    # Get the current Y-axis limits
    ymin, ymax = plt.ylim()
    
    # Generate custom tick positions for Y-axis
    min_power_y = np.floor(np.log10(ymin))
    max_power_y = np.ceil(np.log10(ymax))
    
    # Create tick positions at each power of 10 and halfway between for Y-axis
    tick_positions_y = []
    tick_labels_y = []
    for power in range(int(min_power_y), int(max_power_y) + 1):
        tick_positions_y.append(10**power)
        tick_labels_y.append(f'{int(10**power)}')
        if power < max_power_y:  # Don't add intermediate tick after the last power
            tick_positions_y.append(3 * 10**power)
            tick_labels_y.append(f'{int(3 * 10**power)}')
    
    # Set custom ticks for Y-axis
    plt.yticks(tick_positions_y, tick_labels_y)
    
    # Get the current X-axis limits and generate custom tick positions
    xmin, xmax = plt.xlim()
    min_power_x = np.floor(np.log10(xmin))
    max_power_x = np.ceil(np.log10(xmax))
    
    # Create tick positions at each power of 10 for X-axis, including 0.1 for 0 generations
    tick_positions_x = [0.1]  # Add 0.1 for 0 generations
    tick_labels_x = ['0']     # Label it as 0
    
    for power in range(int(min_power_x), int(max_power_x) + 1):
        if 10**power >= 1:  # Only add powers >= 1 since we already have 0.1
            tick_positions_x.append(10**power)
            tick_labels_x.append(f'{int(10**power)}')
    
    # Set custom ticks for X-axis
    plt.xticks(tick_positions_x, tick_labels_x)
    
    # Add major grid lines
    plt.grid(True, linestyle='--', alpha=0.7, which='major')
    
    # Add minor grid lines
    plt.grid(True, linestyle=':', alpha=0.7, which='minor')
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    
    # Save the second plot
    plot_file = os.path.join(analysis_dir, f"{experiment_name}_connections_vs_generations.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Second plot saved to {plot_file}")
    
    # Create third plot: Neurons vs Generations
    plt.figure(figsize=(10, 8))
    
    # Set both axes to logarithmic scale
    plt.yscale('log')
    plt.xscale('log')
    
    # Plot all points in light blue (using the same generations_for_plot)
    plt.scatter(generations_for_plot, df['neurons'], 
               alpha=0.6, s=120, marker='o')
    
    plt.xlabel('Number of Generations (log scale)')
    plt.ylabel('Number of Neurons (log scale)')
    plt.title(f'Network Size vs Generations - {experiment_name}')
    
    # Get the current Y-axis limits
    ymin, ymax = plt.ylim()
    
    # Generate custom tick positions for Y-axis
    min_power_y = np.floor(np.log10(ymin))
    max_power_y = np.ceil(np.log10(ymax))
    
    # Create tick positions at each power of 10 and halfway between for Y-axis
    tick_positions_y = []
    tick_labels_y = []
    for power in range(int(min_power_y), int(max_power_y) + 1):
        tick_positions_y.append(10**power)
        tick_labels_y.append(f'{int(10**power)}')
        if power < max_power_y:  # Don't add intermediate tick after the last power
            tick_positions_y.append(3 * 10**power)
            tick_labels_y.append(f'{int(3 * 10**power)}')
    
    # Set custom ticks for Y-axis
    plt.yticks(tick_positions_y, tick_labels_y)
    
    # Get the current X-axis limits and generate custom tick positions
    xmin, xmax = plt.xlim()
    min_power_x = np.floor(np.log10(xmin))
    max_power_x = np.ceil(np.log10(xmax))
    
    # Create tick positions at each power of 10 for X-axis, including 0.1 for 0 generations
    tick_positions_x = [0.1]  # Add 0.1 for 0 generations
    tick_labels_x = ['0']     # Label it as 0
    
    for power in range(int(min_power_x), int(max_power_x) + 1):
        if 10**power >= 1:  # Only add powers >= 1 since we already have 0.1
            tick_positions_x.append(10**power)
            tick_labels_x.append(f'{int(10**power)}')
    
    # Set custom ticks for X-axis
    plt.xticks(tick_positions_x, tick_labels_x)
    
    # Add major grid lines
    plt.grid(True, linestyle='--', alpha=0.7, which='major')
    
    # Add minor grid lines
    plt.grid(True, linestyle=':', alpha=0.7, which='minor')
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    
    # Save the third plot
    plot_file = os.path.join(analysis_dir, f"{experiment_name}_neurons_vs_generations.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Third plot saved to {plot_file}")
    
    # Create pivot table: Neurons vs Generations distribution
    # Create bins for neurons and generations to group similar values
    neuron_bins = pd.cut(df['neurons'], bins=10, include_lowest=True)
    generation_bins = pd.cut(df['generations'], bins=10, include_lowest=True)
    
    # Create pivot table
    pivot_table = pd.crosstab(generation_bins, neuron_bins, margins=True, margins_name='Total')
    
    # Save pivot table to CSV
    pivot_file = os.path.join(analysis_dir, f"{experiment_name}_neurons_generations_pivot.csv")
    pivot_table.to_csv(pivot_file)
    print(f"Pivot table saved to {pivot_file}")
    
    # Also create a more detailed version with exact values (no binning)
    # Group by exact neurons and generations values
    detailed_pivot = df.groupby(['generations', 'neurons']).size().unstack(fill_value=0)
    
    # Add totals
    detailed_pivot['Total'] = detailed_pivot.sum(axis=1)
    detailed_pivot.loc['Total'] = detailed_pivot.sum()
    
    # Save detailed pivot table to CSV
    detailed_pivot_file = os.path.join(analysis_dir, f"{experiment_name}_neurons_generations_detailed.csv")
    detailed_pivot.to_csv(detailed_pivot_file)
    print(f"Detailed pivot table saved to {detailed_pivot_file}")
    
    # Create heatmap visualization of the detailed pivot table
    # Remove the 'Total' row and column for visualization
    heatmap_data = detailed_pivot.drop('Total', axis=1).drop('Total', axis=0)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Group generations logarithmically
    generation_values = list(heatmap_data.index)
    
    # Define log-scale bins: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128-255, etc.
    log_bins = [0, 1]
    current_bin = 2
    while current_bin <= max(generation_values):
        next_bin = current_bin * 2 - 1
        log_bins.append((current_bin, next_bin))
        current_bin = next_bin + 1
    
    # Create grouped data
    grouped_data = []
    group_labels = []
    
    for i, bin_def in enumerate(log_bins):
        if isinstance(bin_def, int):
            # Single value bin
            if bin_def in generation_values:
                group_data = heatmap_data.loc[bin_def:bin_def]
                grouped_data.append(group_data)
                group_labels.append(str(bin_def))
        else:
            # Range bin
            start, end = bin_def
            # Use pandas boolean indexing on the index
            mask = (heatmap_data.index >= start) & (heatmap_data.index <= end)
            if mask.any():
                group_data = heatmap_data.loc[mask].sum()
                # Convert Series to DataFrame with proper shape
                group_df = pd.DataFrame([group_data], columns=heatmap_data.columns)
                grouped_data.append(group_df)
                group_labels.append(f"{start}-{end}")
    
    # Combine grouped data
    if grouped_data:
        heatmap_data_grouped = pd.concat(grouped_data, ignore_index=True)
        heatmap_data_grouped.index = group_labels
    else:
        heatmap_data_grouped = heatmap_data
    
    # Use imshow for better control over the visualization
    # Set 0 values to NaN so they appear white
    heatmap_data_plot = heatmap_data_grouped.replace(0, np.nan)
    
    # Create custom colormap that starts from 90% gray instead of white
    colors = [(0.9, 0.9, 0.9), (0.0, 0.0, 0.0)]  # Start from 90% gray, end at black
    custom_cmap = LinearSegmentedColormap.from_list('custom_gray', colors)
    
    im = plt.imshow(heatmap_data_plot, cmap=custom_cmap, aspect='auto', interpolation='nearest', origin='lower')
    
    # Set axis labels
    plt.xlabel('Number of Neurons')
    plt.ylabel('Number of Generations (log groups)')
    plt.title(f'Run Distribution Heatmap - {experiment_name}')
    
    # Set axis ticks
    plt.xticks(range(len(heatmap_data_grouped.columns)), heatmap_data_grouped.columns, rotation=45, ha='right')
    plt.yticks(range(len(heatmap_data_grouped.index)), heatmap_data_grouped.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Number of Runs', rotation=270, labelpad=20)
    
    # Add text annotations for non-zero values
    for i in range(len(heatmap_data_grouped.index)):
        for j in range(len(heatmap_data_grouped.columns)):
            value = heatmap_data_grouped.iloc[i, j]
            if value > 0:  # Only show text for non-zero values
                # Use black font for 1 and 2, white for higher values
                text_color = 'black' if value <= 2 else 'white'
                plt.text(j, i, str(int(value)), ha='center', va='center', 
                        color=text_color, fontsize=20, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_file = os.path.join(analysis_dir, f"{experiment_name}_neurons_generations_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {heatmap_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of runs analyzed: {len(runs_data)}")
    
    # Calculate means and standard deviations
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Mean': means,
        'Std Dev': stds
    })
    
    # Format the output to show mean ± std
    print("\nStatistics (Mean ± Std Dev):")
    for col in summary.index:
        print(f"{col:20}: {summary.loc[col, 'Mean']:10.4f} ± {summary.loc[col, 'Std Dev']:10.4f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results and generate CSV report')
    parser.add_argument('experiment_path', help='Path to experiment directory (e.g., experiments/_ExpB)')
    parser.add_argument('experiment_name', help='Experiment name (e.g., K01_cartpole)')
    parser.add_argument('--no-generations', action='store_true', 
                       help='Disable generations-based coloring and legend (default: show generations)')
    
    args = parser.parse_args()
    
    process_experiment_results(args.experiment_path, args.experiment_name, not args.no_generations)

if __name__ == "__main__":
    main() 