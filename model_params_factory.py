#!/usr/bin/env python3
"""
Model Parameters Factory

Provides clean parameter management for the COIN-SSP pipeline with validation,
filtering, and step-specific parameter creation.
"""

from coin_ssp_core import ModelParams
from typing import Dict, Any


class ModelParamsFactory:
    """
    Factory for creating ModelParams instances with validation and step-specific overrides.

    This class handles:
    - One-time validation and filtering of configuration parameters
    - Clean creation of ModelParams instances for different processing steps
    - Type safety and parameter validation upfront
    """

    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize factory with base parameter configuration.

        Parameters
        ----------
        base_config : Dict[str, Any]
            Raw model_params section from JSON configuration
        """
        self._validated_base = self._validate_and_filter(base_config)

        # Create a test instance to validate all required parameters are present
        try:
            ModelParams(**self._validated_base)
        except Exception as e:
            raise ValueError(f"Invalid base model parameters: {e}")

    def _validate_and_filter(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and filter configuration parameters.

        Parameters
        ----------
        config : Dict[str, Any]
            Raw configuration parameters

        Returns
        -------
        Dict[str, Any]
            Filtered and validated parameters
        """
        # Filter out comment fields (anything starting with '_')
        filtered = {k: v for k, v in config.items() if not k.startswith('_')}

        # Add any additional validation here if needed
        # For example, check parameter ranges, required fields, etc.

        return filtered

    def create_base(self) -> ModelParams:
        """
        Create base ModelParams instance with no modifications.

        Returns
        -------
        ModelParams
            Base parameter instance
        """
        return ModelParams(**self._validated_base)

    def create_for_step(self, step_name: str, **overrides) -> ModelParams:
        """
        Create ModelParams instance for specific processing step with overrides.

        Parameters
        ----------
        step_name : str
            Name of processing step (for documentation/debugging)
        **overrides
            Parameter overrides for this step

        Returns
        -------
        ModelParams
            Parameter instance with step-specific modifications
        """
        params = {**self._validated_base, **overrides}
        return ModelParams(**params)


