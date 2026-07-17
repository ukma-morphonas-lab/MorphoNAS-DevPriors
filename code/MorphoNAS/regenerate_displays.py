#!/usr/bin/env python3
"""
Script to regenerate experiment displays for a given experiment.

This script can be used to regenerate morphogen and neuron graph displays
for experiments that have a best_genome.json file.

Usage:
    python regenerate_displays.py <path_to_experiment>
    
Where <path_to_experiment> can be:
- Path to a best_genome.json file
- Path to a directory containing best_genome.json
- Path to an experiment results directory
"""

import os
import sys
import argparse
from pathlib import Path

# Import necessary functions from main.py
from .main import run_simulation, Genome


def find_best_genome_path(input_path):
    """
    Find the best_genome.json file from the given input path.
    
    Args:
        input_path (str): Path to file or directory
        
    Returns:
        tuple: (best_genome_path, save_dir) where save_dir is the directory to save displays
    """
    input_path = Path(input_path).resolve()
    
    if input_path.is_file():
        # If it's a file, check if it's best_genome.json
        if input_path.name == 'best_genome.json':
            return str(input_path), str(input_path.parent)
        else:
            raise ValueError(f"File {input_path} is not a best_genome.json file")
    
    elif input_path.is_dir():
        # If it's a directory, look for best_genome.json in it
        best_genome_path = input_path / 'best_genome.json'
        if best_genome_path.exists():
            return str(best_genome_path), str(input_path)
        else:
            # Check if it's an experiment results directory structure
            # Look for subdirectories that might contain best_genome.json
            for subdir in input_path.iterdir():
                if subdir.is_dir():
                    potential_best_genome = subdir / 'best_genome.json'
                    if potential_best_genome.exists():
                        return str(potential_best_genome), str(subdir)
            
            raise ValueError(f"No best_genome.json found in {input_path} or its subdirectories")
    
    else:
        raise ValueError(f"Path {input_path} does not exist")





def regenerate_displays(experiment_path, use_kamada_kawai=True, display_node_numbers=False, verbose=False, capture_step=None, capture_steps=None, max_morphogen_cols=3, max_neuron_cols=3, dpi=100, morphogen_display_scale=1.0, neuron_display_scale=1.0, morphogen_multi_step_display_scale=1.0, neuron_multi_step_display_scale=1.0, grayscale=False, font_size=12):
    """
    Regenerate displays for a single experiment.
    
    Args:
        experiment_path (str): Path to experiment (file or directory)
        use_kamada_kawai (bool): Whether to use Kamada-Kawai layout for neuron graph
        display_node_numbers (bool): Whether to display node numbers in neuron graph
        verbose (bool): Whether to run simulation in verbose mode
        capture_step (int, optional): Specific step to capture both morphogen and neuron displays. If None, captures final state.
        capture_steps (list, optional): List of steps to capture in multi-step mode. If provided, overrides capture_step.
        max_morphogen_cols (int): Maximum number of morphogen displays in a row (default: 3)
        max_neuron_cols (int): Maximum number of neuron displays in a row (default: 3)
        dpi (int): DPI (dots per inch) for saved PNG files (default: 100)
        morphogen_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        neuron_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        morphogen_multi_step_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        neuron_multi_step_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        grayscale (bool): Whether to convert all PNG files to grayscale after generation (default: False)
        font_size (int): Base font size in points (default: 12). All text elements will scale relative to this size.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find the best_genome.json file
        best_genome_path, save_dir = find_best_genome_path(experiment_path)
        
        print(f"Loading genome from: {best_genome_path}")
        print(f"Saving displays to: {save_dir}")
        if capture_steps is not None:
            print(f"Capturing displays at steps: {capture_steps}")
        elif capture_step is not None:
            print(f"Capturing displays at step: {capture_step}")
        
        # Load genome and run simulation
        genome = Genome.from_json(filepath=best_genome_path)
        
        # Run simulation and generate displays
        run_simulation(
            genome=genome, 
            verbose=verbose, 
            save_displays=True, 
            save_dir=save_dir, 
            use_kamada_kawai=use_kamada_kawai, 
            display_node_numbers=display_node_numbers,
            capture_step=capture_step,
            capture_steps=capture_steps,
            max_morphogen_cols=max_morphogen_cols,
            max_neuron_cols=max_neuron_cols,
            dpi=dpi,
            morphogen_display_scale=morphogen_display_scale,
            neuron_display_scale=neuron_display_scale,
            morphogen_multi_step_display_scale=morphogen_multi_step_display_scale,
            neuron_multi_step_display_scale=neuron_multi_step_display_scale,
            grayscale=grayscale,
            font_size=font_size
        )
        
        print(f"✓ Successfully regenerated displays for {experiment_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error regenerating displays for {experiment_path}:")
        print(f"  Error: {str(e)}")
        return False


def regenerate_multiple_displays(experiment_paths, use_kamada_kawai=True, display_node_numbers=False, verbose=False, capture_step=None, capture_steps=None, max_morphogen_cols=3, max_neuron_cols=3, dpi=100, morphogen_display_scale=1.0, neuron_display_scale=1.0, morphogen_multi_step_display_scale=1.0, neuron_multi_step_display_scale=1.0, grayscale=False, font_size=12):
    """
    Regenerate displays for multiple experiments.
    
    Args:
        experiment_paths (list): List of paths to experiments
        use_kamada_kawai (bool): Whether to use Kamada-Kawai layout for neuron graph
        display_node_numbers (bool): Whether to display node numbers in neuron graph
        verbose (bool): Whether to run simulation in verbose mode
        capture_step (int, optional): Specific step to capture both morphogen and neuron displays. If None, captures final state.
        capture_steps (list, optional): List of steps to capture in multi-step mode. If provided, overrides capture_step.
        max_morphogen_cols (int): Maximum number of morphogen displays in a row (default: 3)
        max_neuron_cols (int): Maximum number of neuron displays in a row (default: 3)
        dpi (int): DPI (dots per inch) for saved PNG files (default: 100)
        morphogen_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        neuron_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        morphogen_multi_step_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        neuron_multi_step_display_scale (float): Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.
        grayscale (bool): Whether to convert all PNG files to grayscale after generation (default: False)
        font_size (int): Base font size in points (default: 12). All text elements will scale relative to this size.
        
    Returns:
        tuple: (successful_count, failed_count, failed_paths)
    """
    successful = []
    failed = []
    
    for path in experiment_paths:
        print(f"\n{'='*80}")
        print(f"Processing: {path}")
        print(f"{'='*80}")
        
        if regenerate_displays(path, use_kamada_kawai, display_node_numbers, verbose, capture_step, capture_steps, max_morphogen_cols, max_neuron_cols, dpi, morphogen_display_scale, neuron_display_scale, morphogen_multi_step_display_scale, neuron_multi_step_display_scale, grayscale, font_size):
            successful.append(path)
        else:
            failed.append(path)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Display Regeneration Summary")
    print(f"{'='*80}")
    print(f"\nSuccessful regenerations ({len(successful)}):")
    for path in successful:
        print(f"  ✓ {path}")
    
    if failed:
        print(f"\nFailed regenerations ({len(failed)}):")
        for path in failed:
            print(f"  ✗ {path}")
    
    return len(successful), len(failed), failed


def main():
    """Main function to handle command line arguments and execute display regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate experiment displays for given experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate displays for a specific experiment directory
  python regenerate_displays.py experiments/01_nodes_edges_match/results/basic_experiment_02
  
  # Regenerate displays for a specific best_genome.json file
  python regenerate_displays.py experiments/01_nodes_edges_match/results/basic_experiment_02/best_genome.json
  
  # Regenerate displays for multiple experiments
  python regenerate_displays.py exp1 exp2 exp3
  
  # Use different display options
  python regenerate_displays.py --no-kamada-kawai --node-numbers --verbose experiment_path
  
  # Capture displays at a specific step
  python regenerate_displays.py --capture-step 100 experiment_path
  
  # Capture displays at multiple steps (multi-step mode)
  python regenerate_displays.py --capture-steps "10,50,100,200" experiment_path
  
  # Save with high DPI for publication quality
  python regenerate_displays.py --dpi 300 experiment_path
  
  # Save with larger image size (2x scale)
  python regenerate_displays.py --scale 2.0 experiment_path
  
  # Save with high DPI and larger size for publication
  python regenerate_displays.py --dpi 300 --scale 1.5 experiment_path
  
  # Generate displays in grayscale
  python regenerate_displays.py --grayscale experiment_path
  
  # Generate high-quality grayscale displays for publication
  python regenerate_displays.py --grayscale --dpi 300 --scale 1.5 experiment_path
  
  # Use custom font size (e.g., 16pt for larger text)
  python regenerate_displays.py --font-size 16 experiment_path
  
  # Combine custom font size with scaling
  python regenerate_displays.py --font-size 14 --scale 2.0 experiment_path
        """
    )
    
    parser.add_argument(
        'experiment_paths',
        nargs='+',
        help='Path(s) to experiment(s). Can be best_genome.json file, directory containing it, or experiment results directory'
    )
    
    parser.add_argument(
        '--no-kamada-kawai',
        action='store_true',
        help='Disable Kamada-Kawai layout for neuron graph (use default layout)'
    )
    
    parser.add_argument(
        '--node-numbers',
        action='store_true',
        help='Display node numbers in neuron graph'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Run simulation in verbose mode'
    )
    
    parser.add_argument(
        '--capture-step',
        type=int,
        help='Specific step to capture both morphogen and neuron displays (default: final state)'
    )
    
    parser.add_argument(
        '--capture-steps',
        type=str,
        help='Comma-separated list of steps to capture in multi-step mode (e.g., "10,50,100"). Overrides --capture-step if provided.'
    )
    
    parser.add_argument(
        '--max-morphogen-cols',
        type=int,
        default=3,
        help='Maximum number of morphogen displays in a row (default: 3)'
    )
    
    parser.add_argument(
        '--max-neuron-cols',
        type=int,
        default=3,
        help='Maximum number of neuron displays in a row (default: 3)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='DPI (dots per inch) for saved PNG files (default: 100)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor for image size (default: 1.0). Values > 1.0 make images larger, < 1.0 make them smaller.'
    )
    
    parser.add_argument(
        '--scales',
        type=str,
        help='Comma-separated list of scale factors for each display in this order: morphogen, neuron graph, morphogen multi-step, neuron graph multi-step. If not provided, defaults to --scale repeated 4 times. Values > 1.0 make images larger, < 1.0 make them smaller.'
    )
    
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Convert all generated PNG files to grayscale'
    )
    
    parser.add_argument(
        '--font-size',
        type=int,
        default=12,
        help='Base font size in points (default: 12). All text elements will scale relative to this size.'
    )
    
    args = parser.parse_args()
    
    # Set display options
    use_kamada_kawai = not args.no_kamada_kawai
    display_node_numbers = args.node_numbers
    
    print("Experiment Display Regenerator")
    print("="*50)
    print(f"Kamada-Kawai layout: {'enabled' if use_kamada_kawai else 'disabled'}")
    print(f"Node numbers: {'enabled' if display_node_numbers else 'disabled'}")
    print(f"Verbose mode: {'enabled' if args.verbose else 'disabled'}")
    print(f"Grayscale mode: {'enabled' if args.grayscale else 'disabled'}")
    print(f"Max morphogen columns: {args.max_morphogen_cols}")
    print(f"Max neuron columns: {args.max_neuron_cols}")
    print(f"DPI: {args.dpi}")
    print(f"Default scale: {args.scale}")
    
    # Set scales default based on scale if not provided
    if args.scales is None:
        args.scales = ','.join([str(args.scale)] * 4)
    
    print(f"Morphogen display scale: {args.scales.split(',')[0]}")
    print(f"Neuron display scale: {args.scales.split(',')[1]}")
    print(f"Morphogen multi-step display scale: {args.scales.split(',')[2]}")
    print(f"Neuron multi-step display scale: {args.scales.split(',')[3]}")
    print(f"Font size: {args.font_size}")
    # Parse capture_steps if provided
    capture_steps = None
    if args.capture_steps:
        try:
            capture_steps = [int(step.strip()) for step in args.capture_steps.split(',')]
            print(f"Capture steps: {capture_steps}")
        except ValueError:
            print("Error: --capture-steps must be a comma-separated list of integers (e.g., '10,50,100')")
            sys.exit(1)
    else:
        print(f"Capture step: {args.capture_step if args.capture_step is not None else 'final state'}")
    
    print(f"Number of experiments: {len(args.experiment_paths)}")
    print()
    
    scales = args.scales.split(',')
    
    # Regenerate displays
    successful_count, failed_count, failed_paths = regenerate_multiple_displays(
        args.experiment_paths,
        use_kamada_kawai=use_kamada_kawai,
        display_node_numbers=display_node_numbers,
        verbose=args.verbose,
        capture_step=args.capture_step,
        capture_steps=capture_steps,
        max_morphogen_cols=args.max_morphogen_cols,
        max_neuron_cols=args.max_neuron_cols,
        dpi=args.dpi,
        morphogen_display_scale=float(scales[0]),
        neuron_display_scale=float(scales[1]),
        morphogen_multi_step_display_scale=float(scales[2]),
        neuron_multi_step_display_scale=float(scales[3]),
        grayscale=args.grayscale,
        font_size=args.font_size
    )
    
    # Exit with appropriate code
    if failed_count > 0:
        print(f"\nCompleted with {failed_count} failure(s)")
        sys.exit(1)
    else:
        print(f"\nAll {successful_count} experiments processed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main() 