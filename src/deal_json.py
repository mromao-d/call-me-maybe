from pathlib import Path
import json
from .models import schemaPrompt, schemaFunctions, schemaOutput
from pydantic import TypeAdapter, ValidationError
from typing import Any, Union


class ReadJsonException(Exception):
    """Raise Exception when reading file"""
    pass

def _read_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as fd:
            content = json.load(fd)
        return content
    except FileNotFoundError as e:
        raise ReadJsonException(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ReadJsonException(
            f"Invalid JSON formet in {path} (line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e
    except OSError as e:
        raise ReadJsonException(f"Cannot read {path}: {e}") from e
    except Exception as e:
        raise ReadJsonException(f"Error reading {path}: {e}") from e


_PROMPT_ADAPTER = TypeAdapter(list[schemaPrompt])
_FUNC_ADAPTER = TypeAdapter(list[schemaFunctions])
_OUTPUT_ADAPTER = TypeAdapter(list[schemaOutput])

def load_prompt(path: Path):
    try:
        return _PROMPT_ADAPTER.validate_python(_read_json(path))
    except ValidationError as e:
        raise ReadJsonException(f"Invalid prompts in {path} || {e}") from e


def load_funcs(path: str) -> list[schemaFunctions]:
    try:
        return _FUNC_ADAPTER.validate_python(_read_json(path))
    except ValidationError as e:
        raise ReadJsonException(
            f"Ivaldi funcs format is {path} || {e}"
        ) from e


def write_output(path: Path, data: Union[schemaOutput, str]) -> None:
    try:
        if isinstance(data, str):
            validated = _OUTPUT_ADAPTER.validate_json(data)
        else:
            validated = _OUTPUT_ADAPTER.validate_python(data)

        output = _OUTPUT_ADAPTER.dump_json(validated)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('wb') as fd:
            fd.write(output)
    except ValidationError as e:
        raise ReadJsonException(
            f"Ivaldi funcs format is {path} || {e}"
        ) from e