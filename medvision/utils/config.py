"""
Configuration parser module.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def parse_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a YAML configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
