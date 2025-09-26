#!/usr/bin/env python3
"""
Workflow Manager for COIN-SSP Multi-Stage Pipeline

This script orchestrates a three-stage workflow:
1. Stage 1: Run individual response function assessments using initial config
2. Stage 2: Analyze Stage 1 results and generate new config based on GDP-weighted parameter means
3. Stage 3: Run final simulations using the generated config from Stage 2

Usage examples:
    # Full workflow from stage 1
    python workflow_manager.py --config-dir ./configs --stage1-config initial.json

    # Start from stage 2 (if stage 1 already completed)
    python workflow_manager.py --start-stage 2 --stage1-output ./results/stage1_20231201/

    # Only run stage 3
    python workflow_manager.py --start-stage 3 --stage2-config ./configs/generated_config.json
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages the three-stage COIN-SSP workflow pipeline."""

    def __init__(self, config_dir: Path, output_base_dir: Path = None):
        self.config_dir = Path(config_dir)
        self.output_base_dir = output_base_dir or Path('./workflow_outputs')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run_stage1(self, stage1_config: Path, output_dir: Path = None) -> Path:
        """
        Run Stage 1: Individual response function assessments.

        Args:
            stage1_config: Path to initial configuration file
            output_dir: Optional override for output directory

        Returns:
            Path to stage 1 output directory
        """
        if output_dir is None:
            output_dir = self.output_base_dir / f"stage1_{self.timestamp}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Stage 1: Running main.py with config {stage1_config}")
        logger.info(f"Stage 1 output directory: {output_dir}")

        try:
            # Run main.py with the stage 1 config
            cmd = ['python', 'main.py', str(stage1_config)]
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("Stage 1 completed successfully")
            logger.debug(f"Stage 1 stdout: {result.stdout}")

            # The main.py output will be in a timestamped directory
            # We need to find the most recent output directory
            actual_output_dir = self._find_latest_output_directory()
            if actual_output_dir:
                logger.info(f"Stage 1 actual output directory: {actual_output_dir}")
                return actual_output_dir
            else:
                logger.warning("Could not locate Stage 1 output directory, using expected path")
                return output_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Stage 1 failed with exit code {e.returncode}")
            logger.error(f"Stage 1 stderr: {e.stderr}")
            raise

    def _find_latest_output_directory(self) -> Optional[Path]:
        """Find the most recently created output directory from main.py."""
        output_pattern = "output_integrated_*"
        output_dirs = list(Path.cwd().glob(output_pattern))
        if output_dirs:
            return max(output_dirs, key=lambda p: p.stat().st_mtime)
        return None

    def analyze_stage1_results(self, stage1_output_dir: Path) -> Dict[str, Any]:
        """
        Analyze Stage 1 results to extract GDP-weighted parameter means.

        Args:
            stage1_output_dir: Path to Stage 1 output directory

        Returns:
            Dictionary containing analysis results for Stage 2 config generation
        """
        logger.info(f"Starting Stage 2 analysis of: {stage1_output_dir}")

        # This is where you would implement the analysis of Stage 1 results
        # For now, return a placeholder structure
        analysis_results = {
            'gdp_weighted_means': {
                'y_tas1_mean': 0.0,
                'y_tas2_mean': 0.0,
                'k_tas1_mean': 0.0,
                'k_tas2_mean': 0.0,
                'tfp_tas1_mean': 0.0,
                'tfp_tas2_mean': 0.0,
            },
            'stage1_output_dir': str(stage1_output_dir),
            'analysis_timestamp': self.timestamp
        }

        logger.info("Stage 2 analysis completed")
        logger.info(f"GDP-weighted parameter means: {analysis_results['gdp_weighted_means']}")

        return analysis_results

    def generate_stage2_config(self, analysis_results: Dict[str, Any],
                              template_config_path: Path = None,
                              output_config_path: Path = None) -> Path:
        """
        Generate Stage 2 configuration file based on Stage 1 analysis.

        Args:
            analysis_results: Results from analyze_stage1_results
            template_config_path: Optional template config to base Stage 2 config on
            output_config_path: Optional override for output config path

        Returns:
            Path to generated Stage 2 config file
        """
        if output_config_path is None:
            output_config_path = self.config_dir / f"stage2_generated_{self.timestamp}.json"

        output_config_path = Path(output_config_path)
        output_config_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating Stage 2 config: {output_config_path}")

        # Load template config or create a basic structure
        if template_config_path and Path(template_config_path).exists():
            with open(template_config_path, 'r') as f:
                stage2_config = json.load(f)
        else:
            # Create a basic config structure - you'll need to customize this
            stage2_config = {
                "run_metadata": {
                    "json_id": f"stage2_{self.timestamp}",
                    "run_name": f"Stage2_MultiVariable_{self.timestamp}",
                    "description": f"Multi-variable response functions based on Stage 1 analysis",
                    "created_date": datetime.now().strftime('%Y-%m-%d'),
                    "stage1_source": analysis_results.get('stage1_output_dir', 'unknown')
                }
            }

        # Update config with GDP-weighted parameter ratios from analysis
        gdp_means = analysis_results['gdp_weighted_means']

        # Example: Create scaling configurations based on GDP-weighted means
        # This is where you'd implement your specific logic for setting parameter ratios
        if 'response_function_scalings' not in stage2_config:
            stage2_config['response_function_scalings'] = []

        # Add a combined scaling configuration using the ratios
        combined_scaling = {
            "scaling_name": "combined_gdp_weighted",
            "description": f"Combined response using GDP-weighted ratios from Stage 1",
            "y_tas1": gdp_means.get('y_tas1_mean', 0.0),
            "k_tas1": gdp_means.get('k_tas1_mean', 0.0),
            "tfp_tas1": gdp_means.get('tfp_tas1_mean', 0.0)
        }
        stage2_config['response_function_scalings'].append(combined_scaling)

        # Save the generated config
        with open(output_config_path, 'w') as f:
            json.dump(stage2_config, f, indent=2)

        logger.info(f"Stage 2 config generated successfully: {output_config_path}")
        return output_config_path

    def run_stage3(self, stage2_config: Path, output_dir: Path = None) -> Path:
        """
        Run Stage 3: Execute final simulations with generated config.

        Args:
            stage2_config: Path to Stage 2 generated config file
            output_dir: Optional override for output directory

        Returns:
            Path to stage 3 output directory
        """
        if output_dir is None:
            output_dir = self.output_base_dir / f"stage3_{self.timestamp}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Stage 3: Running main.py with generated config {stage2_config}")
        logger.info(f"Stage 3 output directory: {output_dir}")

        try:
            # Run main.py with the stage 2 config
            cmd = ['python', 'main.py', str(stage2_config)]
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("Stage 3 completed successfully")
            logger.debug(f"Stage 3 stdout: {result.stdout}")

            # Find the actual output directory
            actual_output_dir = self._find_latest_output_directory()
            if actual_output_dir:
                logger.info(f"Stage 3 actual output directory: {actual_output_dir}")
                return actual_output_dir
            else:
                return output_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Stage 3 failed with exit code {e.returncode}")
            logger.error(f"Stage 3 stderr: {e.stderr}")
            raise

    def run_full_workflow(self, stage1_config: Path,
                         template_config: Path = None) -> Dict[str, Path]:
        """
        Run the complete three-stage workflow.

        Args:
            stage1_config: Path to initial configuration file
            template_config: Optional template for Stage 2 config generation

        Returns:
            Dictionary with paths to all stage outputs
        """
        logger.info("Starting full three-stage workflow")

        results = {}

        # Stage 1: Individual assessments
        stage1_output = self.run_stage1(stage1_config)
        results['stage1_output'] = stage1_output

        # Stage 2: Analysis and config generation
        analysis_results = self.analyze_stage1_results(stage1_output)
        stage2_config = self.generate_stage2_config(analysis_results, template_config)
        results['stage2_config'] = stage2_config

        # Stage 3: Final simulations
        stage3_output = self.run_stage3(stage2_config)
        results['stage3_output'] = stage3_output

        logger.info("Full workflow completed successfully")
        logger.info(f"Results: {results}")

        return results


def main():
    """Main entry point for the workflow manager."""
    parser = argparse.ArgumentParser(
        description="COIN-SSP Multi-Stage Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # General arguments
    parser.add_argument('--config-dir', type=Path, default='./configs',
                       help='Directory for configuration files (default: ./configs)')
    parser.add_argument('--output-dir', type=Path, default='./workflow_outputs',
                       help='Base directory for workflow outputs (default: ./workflow_outputs)')
    parser.add_argument('--start-stage', type=int, choices=[1, 2, 3], default=1,
                       help='Stage to start workflow from (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    # Stage-specific arguments
    parser.add_argument('--stage1-config', type=Path,
                       default='coin_ssp_config_parameter_sensitivity.json',
                       help='Configuration file for Stage 1 (default: coin_ssp_config_parameter_sensitivity.json)')
    parser.add_argument('--stage1-output', type=Path,
                       help='Stage 1 output directory (required if starting from stage 2+)')
    parser.add_argument('--template-config', type=Path,
                       help='Template configuration for Stage 2 generation')
    parser.add_argument('--stage2-config', type=Path,
                       help='Stage 2 generated config file (required if starting from stage 3)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validation
    if args.start_stage == 1 and not args.stage1_config.exists():
        parser.error(f"Stage 1 config file not found: {args.stage1_config}")
    if args.start_stage >= 2 and not args.stage1_output:
        parser.error("--stage1-output is required when starting from stage 2 or later")
    if args.start_stage == 3 and not args.stage2_config:
        parser.error("--stage2-config is required when starting from stage 3")

    try:
        # Initialize workflow manager
        workflow = WorkflowManager(args.config_dir, args.output_dir)

        if args.start_stage == 1:
            # Full workflow
            results = workflow.run_full_workflow(args.stage1_config, args.template_config)

        elif args.start_stage == 2:
            # Start from stage 2
            analysis_results = workflow.analyze_stage1_results(args.stage1_output)
            stage2_config = workflow.generate_stage2_config(analysis_results, args.template_config)
            stage3_output = workflow.run_stage3(stage2_config)
            results = {
                'stage1_output': args.stage1_output,
                'stage2_config': stage2_config,
                'stage3_output': stage3_output
            }

        elif args.start_stage == 3:
            # Only stage 3
            stage3_output = workflow.run_stage3(args.stage2_config)
            results = {'stage3_output': stage3_output}

        logger.info("Workflow completed successfully!")
        print("\n" + "="*60)
        print("WORKFLOW RESULTS:")
        for stage, path in results.items():
            print(f"  {stage}: {path}")
        print("="*60)

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()