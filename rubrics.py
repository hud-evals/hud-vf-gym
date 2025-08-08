"""Rubrics for HUD Gym."""

from typing import Any

from verifiers import Rubric


class HUDEvaluationRubric(Rubric):
    """Uses HUD's evaluation grade directly.
    
    This is the simplest rubric - it just returns whatever score
    the HUD evaluate tool returned. This is the recommended default
    since HUD's evaluation tools are task-specific and authoritative.
    """
    
    def __init__(self):
        """Initialize with single scoring function."""
        super().__init__(
            funcs=[self.hud_evaluation_score],
            weights=[1.0]
        )
    
    def hud_evaluation_score(self, completion: list[dict[str, Any]], **kwargs) -> float:
        """Extract HUD evaluation score from state.
        
        Args:
            completion: Completion messages (unused)
            **kwargs: Must contain 'state' with 'reward' field
            
        Returns:
            HUD evaluation score (0.0 to 1.0)
        """
        state = kwargs.get("state", {})
        return state.get("reward", 0.0)