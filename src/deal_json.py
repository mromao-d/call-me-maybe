"""Read and validate the input files, and write the output file."""
from pathlib import Path
import json
from .models import schemaPrompt, schemaFunctions, schemaOutput
from pydantic import TypeAdapter, ValidationError
from typing import Any


class ReadJsonException(Exception):
    """Raised when a JSON file cannot be read, parsed or validated."""

    pass


def _read_json(path: Path) -> Any:
    """Read a JSON file and return its content.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        ReadJsonException: If the file is missing, cannot be read or is
            not valid JSON.
    """
    try:
        with path.open('r', encoding='utf-8') as fd:
            content = json.load(fd)
        return content
    except FileNotFoundError as e:
        raise ReadJsonException(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ReadJsonException(
            f"Invalid JSON format in {path} "
            f"(line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e
    except OSError as e:
        raise ReadJsonException(f"Cannot read {path}: {e}") from e
    except Exception as e:
        raise ReadJsonException(f"Error reading {path}: {e}") from e


_PROMPT_ADAPTER = TypeAdapter(list[schemaPrompt])
_FUNC_ADAPTER = TypeAdapter(list[schemaFunctions])
_OUTPUT_ADAPTER = TypeAdapter(list[schemaOutput])


def load_prompt(path: Path) -> list[schemaPrompt]:
    """Load and validate the prompts file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        The validated prompts.

    Raises:
        ReadJsonException: If the file cannot be read, is not valid JSON
            or does not match the expected format.
    """
    try:
        return _PROMPT_ADAPTER.validate_python(_read_json(path))
    except ValidationError as e:
        raise ReadJsonException(f"Invalid prompts in {path} || {e}") from e


def load_funcs(path: Path) -> list[schemaFunctions]:
    """Load and validate the functions definition file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        The validated function definitions.

    Raises:
        ReadJsonException: If the file cannot be read, is not valid JSON
            or does not match the expected format.
    """
    try:
        return _FUNC_ADAPTER.validate_python(_read_json(path))
    except ValidationError as e:
        raise ReadJsonException(
            f"Invalid functions format in {path} || {e}"
        ) from e


def write_output(path: Path, data: list[dict[str, Any]]) -> None:
    """Validate the results and write them to the output file.

    The parent directory is created if it does not exist.

    Args:
        path: Path of the output JSON file.
        data: One dict per answered prompt.

    Raises:
        ReadJsonException: If the results do not match the output format
            or the file cannot be written.
    """
    try:
        validated = _OUTPUT_ADAPTER.validate_python(data)
        output = _OUTPUT_ADAPTER.dump_json(validated, indent=2)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('wb') as fd:
            fd.write(output)
    except ValidationError as e:
        raise ReadJsonException(
            f"Invalid output format for {path} || {e}"
        ) from e
    except OSError as e:
        raise ReadJsonException(f"Cannot write {path}: {e}") from e
