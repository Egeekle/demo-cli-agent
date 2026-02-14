from tests.evals.happy_path_operations.generator import (
    SyntheticDataset,
    SyntheticExample,
    SyntheticExpectations,
    SyntheticInputs,
    generator_agent,
)


def test_synthetic_example_structure():
    """Verify the nested structure is preserved for MLflow."""
    fields = SyntheticExample.model_fields
    assert "inputs" in fields
    assert "expectations" in fields
    assert fields["inputs"].annotation == SyntheticInputs
    assert fields["expectations"].annotation == SyntheticExpectations


def test_expectations_has_required_fields():
    """Verify expectations has all required fields."""
    fields = SyntheticExpectations.model_fields
    assert "operation_type" in fields
    assert "expected_tool" in fields
    assert "expected_appointments" in fields
    assert "expected_answer_contains" in fields


def test_generator_agent_output_schema():
    assert generator_agent.output_type == SyntheticDataset
