"""
Diagnostic and logging framework for warehouse sorting demos.

Easy usage:
    from src.diagnostics import DiagnosticLogger
    
    logger = DiagnosticLogger(model, data, site_name="arm_hand_pinch")
    
    # During operation
    logger.log_state("red_box", "approach", attempt=0)
    
    # At the end
    logger.generate_report()
"""

from .logger import DiagnosticLogger
from .plotter import DiagnosticPlotter
from .metrics import MetricsCalculator

__all__ = ['DiagnosticLogger', 'DiagnosticPlotter', 'MetricsCalculator']

