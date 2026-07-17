import argparse
import json
import os
import random
import glob
from .experiment_runner import ExperimentRunner
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'

def run_experiments(config_dir: str, num_runs: int = 20, experiment_seed: int = None, max_workers: int = None, rerun_on_max_workers_change: bool = False):
    """
    Run multiple experiments for each K*.json config with different seeds.
    
    Args:
        config_dir: Directory containing K*.json configs
        num_runs: Number of runs per config (default: 20)
        experiment_seed: Seed for generating experiment seeds (default: None)
        max_workers: Maximum number of workers for parallel fitness evaluation (default: None)
        rerun_on_max_workers_change: Whether to re-run experiments when max_workers changes (default: False)
    """
    # Set random seed for experiment seeds if provided
    if experiment_seed is not None:
        random.seed(experiment_seed)
    
    # Generate experiment seeds
    experiment_seeds = [random.randint(1, 1000000) for _ in range(num_runs)]
    
    # Get all K*.json configs
    config_files = sorted(glob.glob(os.path.join(config_dir, "K*.json")))
    
    if not config_files:
        print(f"No K*.json configs found in {config_dir}")
        return
    
    print(f"Found {len(config_files)} config files")
    print(f"Will run {num_runs} experiments for each config")
    print(f"Experiment seeds: {experiment_seeds}")
    if max_workers is not None:
        print(f"Max workers: {max_workers}")
    print(f"Rerun on max_workers change: {rerun_on_max_workers_change}")
    
    # Run experiments for each config
    for config_file in config_files:
        # Extract base config name (e.g., "K01" from "K01.json" or "K01_R01.json")
        base_name = os.path.splitext(os.path.basename(config_file))[0]
        if "_R" in base_name:
            base_name = base_name.split("_R")[0]
        
        print(f"\nProcessing {base_name}...")
        
        # Load the config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Run experiments with different seeds
        for run_idx, seed in enumerate(experiment_seeds, 1):
            # Create a copy of the config with the new seed
            run_config = config.copy()
            run_config['ga_params']['seed'] = seed
            
            # Add max_workers to the appropriate section based on optimizer type
            optimizer_type = run_config.get('optimizer_type', 'genetic_algorithm')
            if optimizer_type == 'genetic_algorithm':
                if max_workers is not None:
                    run_config['ga_params']['max_workers'] = max_workers
            elif optimizer_type == 'cma_es':
                if max_workers is not None:
                    if 'cma_params' not in run_config:
                        run_config['cma_params'] = {}
                    run_config['cma_params']['max_workers'] = max_workers
            
            # Create a config file for this run
            run_config_name = f"{base_name}_R{run_idx:02d}"
            run_config_path = os.path.join(config_dir, f"{run_config_name}.json")
            
            # Save the run config
            with open(run_config_path, 'w') as f:
                json.dump(run_config, f, indent=4)
            
            print(f"\nRunning experiment {run_idx}/{num_runs} for {base_name} with seed {seed}")
            
            # Run the experiment
            try:
                runner = ExperimentRunner(run_config_path, rerun_on_max_workers_change=rerun_on_max_workers_change)
                runner.run()
            except Exception as e:
                print(f"Error running experiment: {e}")

def main():
    parser = argparse.ArgumentParser(description='Experiment B runner')
    parser.add_argument('--run-experiments', action='store_true',
                      help='Run experiments for configs')
    parser.add_argument('--config-dir', type=str,
                      default='experiments/_ExpB_RNN_controller/configs',
                      help='Path to the config directory')
    parser.add_argument('--num-runs', type=int, default=20,
                      help='Number of runs per config')
    parser.add_argument('--experiment-seed', type=int, default=None,
                      help='Seed for generating experiment seeds')
    parser.add_argument('--max-workers', type=int, default=None,
                      help='Maximum number of workers for parallel fitness evaluation')
    parser.add_argument('--rerun-on-max-workers-change', action='store_true', default=False,
                      help='Re-run experiments when max_workers changes (default: False, i.e., ignore max_workers changes)')
    
    args = parser.parse_args()
    
    if args.run_experiments:
        run_experiments(args.config_dir, args.num_runs, args.experiment_seed, args.max_workers, args.rerun_on_max_workers_change)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 