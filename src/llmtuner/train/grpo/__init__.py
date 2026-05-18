"""
GRPO (Group Relative Policy Optimization) training module for LLaMA-Factory.

This module integrates TRL's GRPO trainer with LLaMA-Factory's training pipeline,
providing custom reward functions for VMCOT (Visual Multi-modal Chain-of-Thought) training.
"""

from .workflow import run_grpo

__all__ = ["run_grpo"]