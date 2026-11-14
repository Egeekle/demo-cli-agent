import pytest

from src import providers
from tests.evals.conflict_resolution import evaluate as conflict_eval
from tests.evals.happy_path_operations import evaluate as happy_eval
from tests.evals.safety_confirmation import evaluate as safety_eval

# Judges are referenced by (module, attribute) and never bound to a test argument
# or local: a judge's repr includes inference_params (the API key), and pytest
# prints arguments and locals of failing tests.
JUDGES = {
    "conflict_resolution": (conflict_eval, "conflict_judge"),
    "answer_groundedness": (happy_eval, "groundedness_judge"),
    "answer_correctness": (happy_eval, "correctness_judge"),
    "retrieval_recall": (happy_eval, "retrieval_recall_judge"),
    "safety_confirmation": (safety_eval, "safety_judge"),
}


@pytest.mark.parametrize("name", sorted(JUDGES))
def test_judge_uses_shared_endpoint(name):
    module, attr = JUDGES[name]
    wired = (
        getattr(module, attr).model == providers.judge_model_uri
        and getattr(module, attr).inference_params is providers.judge_inference_params
    )
    assert wired, f"{attr} must use providers.judge_model_uri/judge_inference_params"
