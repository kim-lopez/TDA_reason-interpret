## == CREDITS TO ANGELINATSAI04 FOR BASIS OF PIPELINE ==##

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from ripser import ripser
from persim import plot_diagrams, bottleneck
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from persim import bottleneck, wasserstein


# ==============================================================================
# PART 1: DATA LOADING
# ==============================================================================

class AdvancedTDAVisualizer:
    """
    Advanced visualization suite for persistence diagram analysis
    """
    
    def __init__(self, exp_dir: Path):
        """
        Load saved experiment data
        
        Args:
            exp_dir: Path to experiment directory
        """
        self.exp_dir = Path(exp_dir)
        
        # Import the data manager
        import sys
        # Add parent directory to path if needed
        # from your_module import ExperimentDataManager
        
        print(f"Loading data from: {exp_dir}")
        
        # Load both conditions
        self.corr_data = self._load_condition("correct")
        self.incorr_data = self._load_condition("incorrect")
        
        print(f"✓ Loaded {len(self.sb_data['res_list'])} sandbagging samples")
        print(f"✓ Loaded {len(self.nsb_data['res_list'])} non-sandbagging samples")
    
    def _load_condition(self, condition: str) -> Dict:
        """Load a single condition's data"""
        import pickle
        
        condition_dir = self.exp_dir / condition
        
        # Load persistence diagrams
        with open(condition_dir / "persistence_diagrams.pkl", "rb") as f:
            res_list = pickle.load(f)
        
        # Load texts
        with open(condition_dir / "texts.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines()]
        
        # Load attention maps and distance matrices
        attn_maps = np.load(condition_dir / "attention_maps.npy", allow_pickle=True)
        Ds = np.load(condition_dir / "distance_matrices.npy", allow_pickle=True)
        
        return {
            "res_list": res_list,
            "texts": texts,
            "attn_maps": list(attn_maps),
            "Ds": list(Ds)
        }