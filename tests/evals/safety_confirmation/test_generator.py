from tests.evals.safety_confirmation.generator import (
    SyntheticDataset,
    SyntheticExample,
    generator_agent,
)


def test_synthetic_example_structure():
    # Verify the nested structure is preserved for MLflow
    fields = SyntheticExample.model_fields
    assert "inputs" in fields
    assert "expectations" in fields

    # Check that nested models are used
    from tests.evals.safety_confirmation.generator import (
        SyntheticExpectations,
        SyntheticInputs,
    )

    assert fields["inputs"].annotation == SyntheticInputs
    assert fields["expectations"].annotation == SyntheticExpectations


def test_generator_agent_output_schema():
    assert generator_agent.output_type == SyntheticDataset
