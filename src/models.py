from pydantic import BaseModel, ConfigDict
from typing import Literal, Union

types = Literal['number', 'string', 'boolean']
RealTypes = Union[float, str, bool]

class schemaPrompt(BaseModel):
    prompt: str


class schemaDictParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: types


class schemaFunctions(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    description: str
    parameters: dict[str, schemaDictParams]
    returns: schemaDictParams


class schemaOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt: str
    name: str
    parameters: dict[str, RealTypes]
