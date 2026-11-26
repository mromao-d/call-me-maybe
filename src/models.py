"""Pydantic models that validate the input and output JSON files."""
from pydantic import BaseModel, ConfigDict
from typing import Literal, Union

types = Literal['number', 'string', 'boolean']
RealTypes = Union[float, str, bool]


class schemaPrompt(BaseModel):
    """One entry of the prompts file."""

    prompt: str


class schemaDictParams(BaseModel):
    """The type of one parameter (or of the return value)."""

    model_config = ConfigDict(extra='forbid')

    type: types


class schemaFunctions(BaseModel):
    """One function definition from the functions file."""

    model_config = ConfigDict(extra='forbid')

    name: str
    description: str
    parameters: dict[str, schemaDictParams]
    returns: schemaDictParams


class schemaOutput(BaseModel):
    """One function call written to the output file."""

    model_config = ConfigDict(extra='forbid')

    prompt: str
    name: str
    parameters: dict[str, RealTypes]
