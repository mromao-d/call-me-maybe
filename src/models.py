"""Pydantic models that validate the input and output JSON files."""
from pydantic import BaseModel, ConfigDict
from typing import Literal, Union

types = Literal['number', 'integer', 'string', 'boolean']
RealTypes = Union[bool, int, float, str]


class schemaPrompt(BaseModel):
    """One entry of the prompts file.

    Attributes:
        prompt: The natural-language request.
    """

    prompt: str


class schemaDictParams(BaseModel):
    """The type of one parameter, or of a function's return value.

    Attributes:
        type: The JSON type: 'number', 'integer', 'string' or 'boolean'.
    """

    model_config = ConfigDict(extra='forbid')

    type: types


class schemaFunctions(BaseModel):
    """One function definition from the functions file.

    Attributes:
        name: The function name, e.g. 'fn_add_numbers'.
        description: What the function does.
        parameters: Maps each parameter name to its type.
        returns: The type of the return value.
    """

    model_config = ConfigDict(extra='forbid')

    name: str
    description: str
    parameters: dict[str, schemaDictParams]
    returns: schemaDictParams


class schemaOutput(BaseModel):
    """One function call written to the output file.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the chosen function.
        parameters: Maps each parameter name to its value.
    """

    model_config = ConfigDict(extra='forbid')

    prompt: str
    name: str
    parameters: dict[str, RealTypes]
