"""Entry point: uv run python -m src."""
import argparse
import sys
from pathlib import Path
from typing import Any

from .deal_json import _FUNC_ADAPTER, ReadJsonException
from .deal_json import load_funcs, write_output
from .llm import llm


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(
        description="Function calling with a small LLM"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
    )
    return parser.parse_args()


def main() -> None:
    """Load the files, ask the model for every prompt, write the results."""
    args = parse_args()

    try:
        funcs = _FUNC_ADAPTER.dump_python(
            load_funcs(Path(args.functions_definition))
        )
        model = llm()
        prompts = model._validate_prompts(Path(args.input))
    except ReadJsonException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: could not load the model: {e}")
        sys.exit(1)

    result: list[dict[str, Any]] = []
    for prompt in prompts:
        question = prompt['prompt']
        print(f"processing {question}")
        try:
            result.append(model.llm(question, funcs))
        except Exception as e:
            print(f"Error: skipped {question!r}: {e}")

    try:
        write_output(Path(args.output), result)
    except ReadJsonException as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Saved {len(result)} results to {args.output}")


if __name__ == '__main__':
    main()
