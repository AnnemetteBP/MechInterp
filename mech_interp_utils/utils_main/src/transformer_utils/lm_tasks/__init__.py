from ..logit_lens.hooks import make_lens_hooks, clear_lens_hooks
from . import lm_task_manager
from . import lm_task_plotter

from .lm_task_manager import (
    template_to_input_ids,
    cloze_task,
    next_token_pred,
    text_generation_task,
    run_full_analysis,
)
from .lm_task_plotter import plot_full_analysis, plot_full_analysis_grouped

__all__ = [
    'make_lens_hooks',
    'clear_lens_hooks',
    'lm_task_manager',
    'lm_task_plotter',
    'template_to_input_ids',
    'cloze_task',
    'next_token_pred',
    'text_generation_task',
    'run_full_analysis',
    'plot_full_analysis',
    'plot_full_analysis_grouped'
]