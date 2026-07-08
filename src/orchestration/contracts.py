"""Typed artifacts that flow between the agents.

Using structured outputs means each stage hands the next a validated object instead
of free-form prose, which is what keeps the pipeline reliable.
"""

from pydantic import BaseModel, Field


class AcceptanceCriterion(BaseModel):
    id: str = Field(description="e.g. 'AC-1'")
    text: str


class SolutionSpec(BaseModel):
    requirements: list[str]
    acceptance_criteria: list[AcceptanceCriterion]
    data_model: str
    api_contracts: list[str] = Field(
        description=(
            "The EXACT public API the implementation must expose from solution.py, "
            "e.g. 'def update_profile(member_id: str, changes: dict) -> dict'."
        )
    )
    open_questions: list[str] = Field(
        description="Ambiguities a human must resolve. Empty if none."
    )


class CodeFile(BaseModel):
    path: str = Field(description="Relative path, e.g. 'solution.py'.")
    content: str


class Implementation(BaseModel):
    """Developer output."""

    files: list[CodeFile]
    summary: str


class TestSuite(BaseModel):
    """Test-author output."""

    files: list[CodeFile]
    notes: str


class CriterionResult(BaseModel):
    id: str
    passed: bool
    note: str


class QaVerdict(BaseModel):
    """Constructed in code from a real test run — not from an LLM opinion."""

    passed: bool
    results: list[CriterionResult]
    defects: list[str]
    raw_output: str = ""
