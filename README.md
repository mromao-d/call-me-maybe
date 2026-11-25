_This project has been created as part of the 42 curriculum by mromao-s._

# call me maybe

## Description

**call me maybe** turns natural-language requests into structured function calls
using a small language model, **Qwen3-0.6B**.

Given a list of available functions (name, description, typed parameters) and a
list of prompts such as `"What is the sum of 2 and 3?"`, the program asks the
model to choose the right function and fill in its arguments, and writes the
result as JSON:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

A 0.6B model asked to "just answer in JSON" often produces broken or invalid
output. This project uses **constrained decoding** instead: at every generation
step, only the tokens that keep the output valid are allowed, so the result is
always valid JSON that matches the function schema.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install     # uv sync: installs the dependencies and the llm_sdk package
make run         # uv run python -m src
make debug       # runs the program under pdb
make lint        # flake8 + mypy
make clean       # removes caches
```

Running with custom files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Without arguments, the program reads from `data/input/` and writes
`data/output/function_calls.json`.

## Algorithm explanation

### 1. Token texts

The model works with token ids, not with characters. The vocabulary file
(`vocab.json`) stores tokens in a byte-level BPE form, where for example `Ġ`
means a space and `Ċ` means a newline. At start-up the program builds a list
`texts` where `texts[id]` is the real text of each token (using the standard
`bytes_to_unicode` mapping in reverse). This list is what the constraints are
checked against.

### 2. The prompt

The prompt follows Qwen's own tool-calling chat format: a system message with
the function definitions inside `<tools></tools>`, the user question, and the
start of the assistant answer. The assistant part is already filled in up to
`<tool_call>\n{"name": "`, so the model starts directly at the function name.

### 3. Constrained decoding

Generation happens one token at a time. For every step:

1. **Build the allowed list:** check every token of the vocabulary and keep only
   the ones that keep the output valid at this position.
2. **Mask:** get the model's scores (logits) for all tokens, set every token that
   is not allowed to `-inf`, and take the highest remaining score (`argmax`).

The model therefore still makes every choice, but only among valid options.

Different parts of the JSON use different rules:

| Part          | Rule for an allowed token                                                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Function name | `written + token` must still be the start of some `function_name"`                                                                                        |
| Number        | `written + token` must still be the start of a valid number (`-`, digits, one `.`); once the number is complete, a token starting with `,` or `}` ends it |
| String        | no `"`, no newline; a token starting with `"` closes the string                                                                                           |
| Boolean       | same as the function name, with the options `true` / `false`                                                                                              |

### 4. Fixed text

Everything that is always the same is not generated: `", "parameters": {"`,
the parameter names, commas and quotes are encoded and appended directly
(`_add_manual_text`). This saves model calls and guarantees the structure of
the JSON. The model is only asked for the parts it must decide: the function
name and the values.

## Design decisions

- **Character-level checks on decoded token texts** instead of a full JSON
  grammar or state machine: the output format is always the same
  (`name` + flat `parameters`), so a few simple rules are enough and easy to
  explain.
- **Pre-filling the fixed JSON parts** instead of letting the model write them:
  fewer steps (faster) and no possibility of a malformed structure.
- **The prompt is added to the output by the program**, not generated: it is
  already known, so asking the model to copy it would only add risk.
- **pydantic models** (`src/models.py`) validate the input files and the output
  before writing, so invalid input gives a clear error instead of a crash.
- **One prompt at a time:** the SDK has no batching and no KV cache, so each
  question is processed on its own.

## Performance analysis

Measured on the provided test set (11 prompts) on a MacBook:

- **Valid JSON:** 100% — guaranteed by construction, every value goes through a
  constrained step and the result is validated with pydantic.
- **Correct function chosen:** 11 / 11.
- **Fully correct calls (function + all arguments):** 10 / 11 (91%). The one
  failure is the "replace all numbers" regex prompt (see Challenges).
- **Time:** about 65–70 seconds for the whole set (limit: 5 minutes), plus a few
  seconds to load the model.

Most of the time is spent in the model itself: the SDK has no KV cache, so every
new token re-processes the whole prompt. Building the allowed list scans the
~151k-token vocabulary in Python at every step, which is simple but could be
precomputed for the rules that never change.

## Challenges

**Understanding the tokens.** The vocabulary does not contain plain text, but a
byte-level encoding (`Ġ` for a space, `Ċ` for a newline, multi-byte characters
split across tokens). The constraints only work after converting every token to
its real text.

**Constraints guarantee valid output, not correct output.** This was the main
difficulty with the regex prompts. For
`Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS`
the model writes `regex: "NUMBERS"` instead of a real pattern such as `\d+`.

- First suspicion: string values did not allow the backslash character, so the
  model could not write `\d+`. Allowing backslashes (and decoding JSON escapes
  such as `\\d` into `\d` at the end, while making sure an escaped `\"` does not
  close the string) did **not** change the result.
- Printing the model's top scores at the first regex token showed the real
  reason: `numbers` and `([` (the start of a real regex) are tied at 21.11,
  followed by `3`, `NUM` and `(\`. The 0.6B model is simply unsure, and the
  constraint can only choose among valid options — it cannot make the model
  pick the _right_ one.
- Adding a sentence to the system prompt ("a regex argument must be a real
  regular expression pattern") moved `([` to first place when probing a single
  step, but the full run still produced `NUMBERS`: the manually appended text
  and the generated text are not split into tokens exactly the same way as the
  probe, so the context the model sees is slightly different.

- Writing the opening quote of a string value in the same piece as the
  parameter name (`"regex": "` instead of `"regex": ` followed by `"`), so the
  model sees the same tokens as in normal JSON, also did not fix it. Each small
  change only moved the model to another wrong answer (`NUMBERS`, `34|233`,
  `NUMBERS` again).
- A system-prompt rule that mentions `\d+` made the model write `\d+`, but it
  then repeated `|\d+|\d+|...` until the length limit — and the hint almost
  gives away the answer for this exact prompt, which would be overfitting the
  test set. A worked example with a different regex task did not help. This
  prompt is still wrong: no general fix was found.

**Spaces at the start of string values (solved).** For
`Replace all vowels in 'Programming is fun' with asterisks`, the regex
`([aeiouAEIOU])` was correct but the replacement was wrong. The steps:

1. The model first wrote `**` instead of `*`.
2. After the quote fix above it wrote `" *"`: it chose the token ` *`, which
   starts with a space. In byte-level BPE, `*` and ` *` are different tokens,
   and right after an opening quote the model preferred the one with the space.
3. Banning tokens that start with a space at the beginning of a value made it
   worse: the model switched to the word `asterisk`.
4. **Solution:** let the model choose freely, and remove spaces from the start
   and end of the finished value (`value.strip()`). The model's real choice
   was the symbol `*`; the space was only a side effect of how the text is split
   into tokens. With this change the result is `*`, and the other prompts
   are not affected.

Trade-off: a value that really starts or ends with a space (for example
"replace with a space") would lose that space. This is rare in function
arguments, so removing tokenizer spaces was the better choice.

Result on the provided test set: **10 / 11 fully correct (91%)**, 11 / 11
correct function names.

## Testing strategy

- Running the full provided test set and checking every output by hand
  (function name and each argument).
- Printing the top-scoring tokens at a given position to understand _why_ the
  model makes a choice (used for the regex problem above).
- Testing the helper functions alone (for example `_is_number` with `"3."`,
  `"-"`, `"12"`, `"1.5"`), which found a crash on `"-"` and a bug that made
  decimals impossible.
- Error cases: missing file, invalid JSON, wrong schema in the input files.
- `make lint` (flake8 + mypy) with no errors.

## Example usage

```bash
$ uv run python -m src
$ cat data/output/function_calls.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  },
  {
    "prompt": "What is the square root of 16?",
    "name": "fn_get_square_root",
    "parameters": { "a": 16.0 }
  }
]
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen function calling documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Qwen key concepts (tokens, chat format)](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [Qwen 3 chat template deep dive (Hugging Face blog)](https://github.com/huggingface/blog/blob/main/qwen-3-chat-template-deep-dive.md)
- [Qwen 3 0.6B: why small models still matter](https://medium.com/@marketing_novita.ai/qwen-3-0-6b-why-small-models-still-matter-today-bb593d3efaa9)
- [Prompt injection detection via fine-tuning (example project)](https://github.com/ItayMoh/Prompt-Injection-Detection-via-Fine-Tuning)

### AI usage

Claude (Anthropic) was used as a tutor during this project: to explain
byte-level BPE tokenization and constrained decoding, to review my code and
point out bugs (for example the wrong quote character in the function-name
targets, encoding the question instead of the full prompt, and the number
parsing bugs), to investigate the regex difficulty described above by printing
the model's top scores, and to help draft this README. The structure of the
program and the code are my own, and I can explain every part of it.
