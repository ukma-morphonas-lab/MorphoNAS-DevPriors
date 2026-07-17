import sys
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

# Suppress warnings for clarity
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = ArgumentParser(description='Analyze Experiment A results.')
    parser.add_argument('results_path', type=str, help='Path to the experiment results directory.')
    parser.add_argument('--output', type=str, default='analysis_results', help='Path to save analysis outputs.')
    parser.add_argument('--genome-table', type=str, help='Path to specific experiment directory to generate genome LaTeX table.')
    parser.add_argument('--svg-output', action='store_true', help='Generate SVG visualization of genome parameters.')
    return parser.parse_args()

def load_data(results_path):
    records = []

    for folder in glob.glob(os.path.join(results_path, 'K*_R*')):
        k_r = os.path.basename(folder).split('_')
        K = int(k_r[0][1:])
        R = int(k_r[1][1:])

        stats_file = os.path.join(folder, 'stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                data = json.load(f)

            targets = data.get('configuration', {}).get('fitness_targets', {})

            records.append({
                'K': K,
                'R': R,
                'generations': data['generations'],
                'evaluations': data['evaluations'],
                'best_fitness': data['best_fitness'],
                'elapsed_time': data['elapsed_time'],
                'target_neurons': targets.get('neurons'),
                'target_inputs': targets.get('no_incoming'),
                'target_connections': targets.get('connections')
            })

    df = pd.DataFrame(records)
    return df

def analyze_data(df, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Calculate summary statistics including targets
    summary = df.groupby('K').agg(
        mean_generations=('generations', 'mean'),
        std_generations=('generations', 'std'),
        mean_evaluations=('evaluations', 'mean'),
        std_evaluations=('evaluations', 'std'),
        mean_elapsed_time=('elapsed_time', 'mean'),
        success_rate=('best_fitness', lambda x: np.mean(x >= 1.0)),
        target_neurons=('target_neurons', 'first'),
        target_inputs=('target_inputs', 'first'),
        target_connections=('target_connections', 'first')
    ).reset_index()

    # Add label with integer values, handling float issue
    summary['K_label'] = summary.apply(
        lambda row: f"K{int(row['K']):02d}", axis=1
    )

    summary.to_csv(os.path.join(output_path, 'summary_statistics.csv'), index=False)

    # Visualization for generations
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='target_neurons', y='target_connections', size='mean_generations', hue='target_inputs', data=summary, palette='viridis', sizes=(50, 400))
    plt.title('Generations to Convergence by Graph Complexity')
    plt.xlabel('Target Neurons')
    plt.ylabel('Target Connections')
    plt.legend(title='Target Inputs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_path, 'generations_complexity_scatter.png'), bbox_inches='tight')
    plt.close()

    # Generate LaTeX table instead of barplot
    latex_table = r"""\begin{table}[ht]
\centering
\caption{Success rates across target sets sampled from random graphs.}
\label{tab:success_rates}
\begin{tabular}{lccc|c}
\toprule
\textbf{Target Set} & \(N_{\text{target}}\) & \(E_{\text{target}}\) & \(S_{\text{target}}\) & \textbf{Success Rate (\(R=20\))} \\
\midrule
"""
    
    # Sort by K value for consistent ordering
    summary_sorted = summary.sort_values('K')
    
    for _, row in summary_sorted.iterrows():
        k_label = f"K{int(row['K']):02d}"
        n_target = int(row['target_neurons'])
        e_target = int(row['target_connections'])
        s_target = int(row['target_inputs'])
        success_rate = f"{row['success_rate']:.2f}"
        
        latex_table += f"{k_label} & {n_target} & {e_target} & {s_target} & {success_rate} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    # Save LaTeX table to file
    with open(os.path.join(output_path, 'success_rates_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print(f'LaTeX table saved to {os.path.join(output_path, "success_rates_table.tex")}')
    print(f'Analysis completed. Results saved to {output_path}')

def generate_genome_table(experiment_path, output_path):
    """Generate LaTeX table from best_genome.json"""
    genome_file = os.path.join(experiment_path, 'best_genome.json')
    
    if not os.path.exists(genome_file):
        print(f"Error: best_genome.json not found in {experiment_path}")
        return
    
    with open(genome_file, 'r') as f:
        genome = json.load(f)
    
    # Extract experiment identifier (e.g., K17 from path)
    exp_name = os.path.basename(experiment_path)
    exp_name_escaped = exp_name.replace('_', '\\_')
    
    # Format arrays for secretion rates
    progenitor_rates = ', '.join([f'{x:.3f}' for x in genome['progenitor_secretion_rates']])
    neuron_rates = ', '.join([f'{x:.3f}' for x in genome['neuron_secretion_rates']])
    
    # Build LaTeX table in parts to avoid f-string issues
    latex_parts = []
    latex_parts.append(r"\begin{table}[ht]")
    latex_parts.append(r"\centering")
    latex_parts.append(f"\\caption{{Genome \\(G\\) of a successful individual evolved for the {exp_name_escaped} target. The genome encodes spatial, chemical, and behavioral parameters used by the developmental automaton.}}")
    latex_parts.append(f"\\label{{tab:genome_{exp_name.lower()}}}")
    latex_parts.append(r"\renewcommand{\arraystretch}{1.2}")
    latex_parts.append(r"\begin{tabular}{ll}")
    latex_parts.append(r"\toprule")
    latex_parts.append(r"\textbf{Component} & \textbf{Values} \\")
    latex_parts.append(r"\midrule")
    latex_parts.append(f"\\textbf{{G\\textsubscript{{dim}}}} & \\(L_x = {genome['size_x']},\\quad L_y = {genome['size_y']}\\) \\\\")
    latex_parts.append(f"\\textbf{{G\\textsubscript{{iter}}}} & {genome['max_growth_steps']} \\\\")
    latex_parts.append(r"\midrule")
    latex_parts.append(r"\textbf{G\textsubscript{morph}}: Morphogens & 3 types \\")
    latex_parts.append(f"\\quad Progenitor secretion rates & [{progenitor_rates}] \\\\")
    latex_parts.append(f"\\quad Neuron secretion rates & [{neuron_rates}] \\\\")
    
    # Inhibition matrix
    latex_parts.append(r"\quad Inhibition matrix &")
    latex_parts.append(r"\(\begin{bmatrix}")
    latex_parts.append(f"{genome['inhibition_matrix'][0][0]:.3f} & {genome['inhibition_matrix'][0][1]:.3f} & {genome['inhibition_matrix'][0][2]:.3f} \\\\")
    latex_parts.append(f"{genome['inhibition_matrix'][1][0]:.3f} & {genome['inhibition_matrix'][1][1]:.3f} & {genome['inhibition_matrix'][1][2]:.3f} \\\\")
    latex_parts.append(f"{genome['inhibition_matrix'][2][0]:.3f} & {genome['inhibition_matrix'][2][1]:.3f} & {genome['inhibition_matrix'][2][2]:.3f}")
    latex_parts.append(r"\end{bmatrix}\) \\")
    
    # Diffusion coefficients
    latex_parts.append(r"\quad Diffusion coefficients &")
    latex_parts.append(r"\begin{tabular}[t]{@{}l@{}}")
    
    for i in range(3):
        latex_parts.append(f"Morphogen {i+1}: ")
        latex_parts.append(r"\(\begin{bmatrix}")
        latex_parts.append(f"{genome['diffusion_patterns'][i][0][0]:.3f} & {genome['diffusion_patterns'][i][0][1]:.3f} & {genome['diffusion_patterns'][i][0][2]:.3f} \\\\")
        latex_parts.append(f"{genome['diffusion_patterns'][i][1][0]:.3f} & {genome['diffusion_patterns'][i][1][1]:.3f} & {genome['diffusion_patterns'][i][1][2]:.3f} \\\\")
        latex_parts.append(f"{genome['diffusion_patterns'][i][2][0]:.3f} & {genome['diffusion_patterns'][i][2][1]:.3f} & {genome['diffusion_patterns'][i][2][2]:.3f}")
        latex_parts.append(r"\end{bmatrix}\) \\")
    
    latex_parts.append(r"\end{tabular} \\")
    latex_parts.append(r"\midrule")
    latex_parts.append(r"\textbf{G\textsubscript{fates}}: Behavioral thresholds & \\")
    latex_parts.append(f"\\quad Cell division & {genome['division_threshold']:.3f} \\\\")
    latex_parts.append(f"\\quad Differentiation & {genome['cell_differentiation_threshold']:.3f} \\\\")
    latex_parts.append(f"\\quad Axon growth & {genome['axon_growth_threshold']:.3f} \\\\")
    latex_parts.append(r"\midrule")
    latex_parts.append(r"\textbf{G\textsubscript{axon}}: Axon parameters & \\")
    latex_parts.append(f"\\quad Connection threshold & {genome['axon_connect_threshold']:.3f} \\\\")
    latex_parts.append(f"\\quad Max axon length & {genome['max_axon_length']} \\\\")
    latex_parts.append(f"\\quad Self-connect isolated neurons & {genome['self_connect_isolated_neurons_fraction']:.3f} \\\\")
    latex_parts.append(r"\midrule")
    latex_parts.append(r"Other parameters & \\")
    latex_parts.append(f"\\quad Diffusion rate (global) & {genome['diffusion_rate']:.3f} \\\\")
    latex_parts.append(f"\\quad Weight adjustment target & {genome['weight_adjustment_target']:.3f} \\\\")
    latex_parts.append(f"\\quad Weight adjustment rate & {genome['weight_adjustment_rate']:.3f} \\\\")
    latex_parts.append(r"\bottomrule")
    latex_parts.append(r"\end{tabular}")
    latex_parts.append(r"\end{table}")
    
    latex_table = '\n'.join(latex_parts)
    
    # Save LaTeX table to file
    output_file = os.path.join(output_path, f'genome_table_{exp_name.lower()}.tex')
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f'Genome LaTeX table saved to {output_file}')

def generate_genome_svg(experiment_path, output_path):
    import os, json

    genome_file = os.path.join(experiment_path, 'best_genome.json')
    if not os.path.exists(genome_file):
        print(f"Error: best_genome.json not found in {experiment_path}")
        return

    with open(genome_file, 'r') as f:
        genome = json.load(f)

    # Settings
    svg_width = 1000
    svg_height = 800
    line_height = 20
    font_size = 14
    start_x, start_y = 20, 40
    num_morphogens = 3

    # --- Diffusion matrices ---
    matrix_lines = []
    for i in range(num_morphogens):
        pattern = genome['diffusion_patterns'][i]
        pattern_width = 5 * len(pattern[0]) + (len(pattern[0]) - 1)
        header_width = max(pattern_width, 17)
        lines = [f"{'D_M' + str(i):^{header_width}}"]
        for row in pattern:
            lines.append(" ".join([f"{x:.3f}" for x in row]))
        matrix_lines.append(lines)

    diffusion_lines = []
    max_lines = max(len(lines) for lines in matrix_lines)
    for line_idx in range(max_lines):
        line_parts = []
        for m, matrix in enumerate(matrix_lines):
            pattern = genome['diffusion_patterns'][m]
            pattern_width = 5 * len(pattern[0]) + (len(pattern[0]) - 1)
            header_width = max(pattern_width, 17)
            if line_idx < len(matrix):
                line_parts.append(f"{matrix[line_idx]:{header_width}}")
            else:
                line_parts.append(" " * header_width)
        diffusion_lines.append("    ".join(line_parts))
#    diffusion_lines.append("")  # Extra spacing

    # --- Inhibition + Thresholds + Secretion ---
    matrix_width = 4 + 6 * num_morphogens
    header_spacing = 8
    label_spacing = header_spacing + 2
    header_row_spacing = label_spacing + 3
    secretion_spacing = 20

    inhib_lines = []
    inhib_lines.append(f"{'Inhibition:':^{matrix_width}}" + " " * header_spacing + "Gfates:")
    inhib_lines.append("   " + "    ".join([f"M{i}" for i in range(num_morphogens)])
                    + " " * header_row_spacing + f"Division:           {genome['division_threshold']:.3f}")

    for i, row in enumerate(genome['inhibition_matrix']):
        line = f"M{i} " + " ".join([f"{x:.3f}" for x in row])
        if i == 0:
            line += " " * label_spacing + f"Differentiation:    {genome['cell_differentiation_threshold']:.3f}"
            line += " " * (secretion_spacing - 14)
        elif i == 1:
            line += " " * label_spacing + f"Axon Growth:        {genome['axon_growth_threshold']:.3f}"
        inhib_lines.append(line)

    # Spacing
    inhib_lines.append("")  # One line break before metadata

    # Add secretion rates to the bottom
    inhib_lines.append("Secretion Rates:")
    inhib_lines.append(f"  Progenitor: {', '.join([f'{x:.3f}' for x in genome['progenitor_secretion_rates']])}")
    inhib_lines.append(f"  Neuron:     {', '.join([f'{x:.3f}' for x in genome['neuron_secretion_rates']])}")
    
    # Add G_axon block to the right of Secretion Rates
    # Calculate spacing to align G_axon with Secretion Rates
    secretion_label_width = len("Secretion Rates:")
    spacing_to_align = secretion_label_width + 8  # Reduced spacing to move G_axon closer
    
    # Calculate actual width of secretion rates lines for proper alignment
    progenitor_line_width = len(f"  Progenitor: {', '.join([f'{x:.3f}' for x in genome['progenitor_secretion_rates']])}")
    neuron_line_width = len(f"  Neuron:     {', '.join([f'{x:.3f}' for x in genome['neuron_secretion_rates']])}")
    max_secretion_width = max(progenitor_line_width, neuron_line_width)
    
    # Calculate spacing for each line based on actual content width
    spacing_for_progenitor = max_secretion_width - progenitor_line_width + 6
    spacing_for_neuron = max_secretion_width - neuron_line_width + 6
    
    # Update the secretion rates lines to include G_axon on the same lines
    inhib_lines[-3] = f"Secretion Rates:{' ' * spacing_to_align}G_axon:"
    inhib_lines[-2] = f"  Progenitor: {', '.join([f'{x:.3f}' for x in genome['progenitor_secretion_rates']])} {' ' * spacing_for_progenitor}Axon Connect: {genome['axon_connect_threshold']:.3f}"
    inhib_lines[-1] = f"  Neuron:     {', '.join([f'{x:.3f}' for x in genome['neuron_secretion_rates']])} {' ' * spacing_for_neuron}Max Axon Length: {genome['max_axon_length']}"

    # --- Combine lines ---
    all_lines = diffusion_lines + [""] + inhib_lines

    # --- Add metadata at the top ---
    # Extract data
    progenitor = " ".join([f"{x:.5f}" for x in genome['progenitor_secretion_rates']])
    neuron = " ".join([f"{x:.5f}" for x in genome['neuron_secretion_rates']])
    steps = genome.get("max_growth_steps", "N/A")
    lattice_str = f"{genome['size_x']} x {genome['size_y']}"
    
    # Create metadata lines for the top
    metadata_lines = [
        f"G_iter: {steps:<10}  Lattice Size: {lattice_str}",
        ""  # Empty line for spacing
    ]
    
    # Combine all lines with metadata at top
    all_lines = metadata_lines + all_lines

    # --- Generate SVG ---
    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        f'    .text {{ font-family: monospace; font-size: {font_size}px; white-space: pre; }}',
        '    .subscript { font-size: 0.8em; baseline-shift: sub; }',
        '  </style>'
    ]

    for idx, line in enumerate(all_lines):
        y = start_y + idx * line_height
        # Check if line contains "Gfates:" and format it with subscript
        if "Gfates:" in line:
            # Split the line to handle the subscript formatting
            parts = line.split("Gfates:")
            if len(parts) == 2:
                svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{parts[0]}G<tspan class="subscript">fates</tspan>:{parts[1]}</text>')
            else:
                svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{line}</text>')
        # Check if line contains "G_iter:" and format it with subscript
        elif "G_iter:" in line:
            # Split the line to handle the subscript formatting
            parts = line.split("G_iter:")
            if len(parts) == 2:
                svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{parts[0]}G<tspan class="subscript">iter</tspan>:{parts[1]}</text>')
            else:
                svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{line}</text>')
        # Check if line contains "G_axon:" and format it with subscript
        elif "G_axon:" in line:
            # Split the line to handle the subscript formatting
            parts = line.split("G_axon:")
            if len(parts) == 2:
                svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{parts[0]}G<tspan class="subscript">axon</tspan>:{parts[1]}</text>')
            else:
                svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{line}</text>')
        # Check if line contains "D_M" headers and format them with subscript
        elif "D_M" in line:
            # Find all occurrences of D_M followed by a number
            import re
            formatted_line = re.sub(r'D_M(\d+)', r'D<tspan class="subscript">M\1</tspan>', line)
            svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{formatted_line}</text>')
        else:
            svg_parts.append(f'  <text x="{start_x}" y="{y}" class="text">{line}</text>')

    svg_parts.append('</svg>')

    # Save SVG
    exp_name = os.path.basename(experiment_path)
    output_file = os.path.join(output_path, f'genome_text_{exp_name.lower()}.svg')
    with open(output_file, 'w') as f:
        f.write("\n".join(svg_parts))

    print(f"SVG generated at: {output_file}")

def main():
    args = parse_arguments()

    df = load_data(args.results_path)
    analyze_data(df, args.output)
    
    if args.genome_table:
        generate_genome_table(args.genome_table, args.output)
        if args.svg_output:
            generate_genome_svg(args.genome_table, args.output)

if __name__ == '__main__':
    main()
