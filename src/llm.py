"""Function calling with Qwen3-0.6B using constrained decoding."""
import json
from llm_sdk import Small_LLM_Model
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import numpy as np
from .deal_json import load_prompt, _PROMPT_ADAPTER
from typing import Any


class llm(BaseModel):
    """Wraps the model and generates one function call per question."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = None
    vocab: dict[str, int] = {}
    byte_decoder: dict[str, int] = {}
    vocab_size: int = 0
    texts: list[str] = []
    string_start: list[int] = []

    def model_post_init(self, context: Any) -> None:
        """Load the model and build the text of every token."""
        self.model = Small_LLM_Model()

        vocab_path = self.model.get_path_to_vocab_file()
        self.vocab = self._read_json(vocab_path)
        self.byte_decoder = self._byte_decoder()
        self._id_to_token()
        self.string_start = self._string_start_tokens()

    def _string_start_tokens(self) -> list[int]:
        """Tokens that open a JSON string: '"' or ' "' followed by content."""
        allowed = []
        for token_id, text in enumerate(self.texts):
            for prefix in (' "', '"'):
                if text.startswith(prefix):
                    rest = text[len(prefix):]
                    if ("\n" not in rest
                            and self._first_unescaped_quote(rest, False)
                            == -1):
                        allowed.append(token_id)
                    break
        return allowed

    def _bytes_to_unicode(self) -> dict[int, str]:
        """
        Map every byte (0-255) to the printable character used in vocab.json.

        Bytes that are already printable map to themselves. The others
        (spaces, control characters) are moved to characters above 256,
        for example a space becomes 'Ġ' and a newline becomes 'Ċ'.
        """

        printable: list[int] = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        mapping: dict[int, str] = {b: chr(b) for b in printable}
        extra = 0

        for b in range(256):
            if b not in mapping:
                mapping[b] = chr(256 + extra)
                extra += 1

        return mapping

    def _byte_decoder(self) -> dict[str, int]:
        """Reverse mapping: printable character -> original byte."""
        return {char: byte for byte, char in self._bytes_to_unicode().items()}

    def _token_to_printable(self, token: str) -> str:
        """Turn a vocab.json token into the real text it stands for."""
        raw = bytes(self.byte_decoder[c] for c in token)
        return raw.decode("utf-8", errors="replace")

    def _id_to_token(self) -> None:
        """Build self.texts, where texts[id] is the real text of token id."""
        self.vocab_size = len(self.model.get_logits_from_input_ids([0]))
        self.texts = [""] * self.vocab_size
        for token, tid in self.vocab.items():
            if tid < self.vocab_size:
                self.texts[tid] = self._token_to_printable(token)

    def _next_token(self, ids: list[int], allowed: list[int]) -> int:
        """
        Ask the model for the next token, among the allowed ones only.

        Every token that is not allowed gets a score of -inf,
        so argmax can only pick an allowed token.
        """
        scores = np.asarray(self.model.get_logits_from_input_ids(ids))
        masked = np.full(self.vocab_size, -np.inf)
        masked[allowed] = scores[allowed]
        return int(np.argmax(masked))

    def _chose_function(
        self, ids: list[int], names: list[str], end: str = '"'
    ) -> str:
        """
        Let the model write one of the names, one token at a time.

        A token is allowed only if the text stays the start of some name.
        """
        targets = [name + end for name in names]

        written = ""
        while written not in targets:
            allowed = []
            for token_id, text in enumerate(self.texts):
                if text == "":
                    continue
                new = written + text
                for target in targets:
                    if target.startswith(new):
                        allowed.append(token_id)
                        break
            if len(allowed) == 0:
                raise ValueError("Could not choose a function")

            token_id = self._next_token(ids, allowed)
            ids.append(token_id)
            written = written + self.texts[token_id]

        return written[:len(written) - len(end)]

    def _add_manual_text(self, ids: list[int], text: str) -> None:
        """
        Append fixed text without asking the model.

        Saves model calls and guarantees the JSON structure.
        """
        ids.extend(self.model.encode(text)[0].tolist())

    def _is_number(
        self, text: str, full: bool = False, integer: bool = False
    ) -> bool:
        """
        Check a number.

        full=False: text can still become a number ('-', '12', '3.').
        full=True: text is a complete number ('12', '-3.5').
        integer=True: no decimal point is allowed ('12', '-3').
        """
        if integer and "." in text:
            return False
        if full and (text in ("", "-") or text.endswith(".")):
            return False
        if text.startswith('-'):
            text = text[1:]
        if text.startswith('-') or text.startswith('.'):
            return False
        if text.count(".") > 1:
            return False
        for char in text:
            if char not in "0123456789.":
                return False
        return True

    def _add_number(
        self, ids: list[int], integer: bool = False
    ) -> int | float:
        """
        Let the model write a number.

        Returns an int if integer=True, otherwise a float.
        """
        writen = ""
        for _ in range(20):
            allowed = []
            for token_id, text in enumerate(self.texts):
                if text == "":
                    continue
                if self._is_number(writen + text, integer=integer):
                    allowed.append(token_id)
                elif (self._is_number(writen, True, integer)
                      and text[0] in ",}"):
                    allowed.append(token_id)

            token_id = self._next_token(ids, allowed)
            text = self.texts[token_id]
            if text[0] in ",}":
                break
            ids.append(token_id)
            writen = writen + text

        if not self._is_number(writen, True, integer):
            raise ValueError(f"Bad number: {writen!r}")
        return int(writen) if integer else float(writen)

    def _ends_with_escape(self, text: str) -> bool:
        """True if text ends with an unfinished backslash escape."""
        count = 0
        for char in reversed(text):
            if char != "\\":
                break
            count += 1
        return count % 2 == 1

    def _first_unescaped_quote(self, text: str, escape: bool) -> int:
        """Index of the first '"' in text that is not escaped, or -1."""
        for i, char in enumerate(text):
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                return i
        return -1

    def _add_string(self, ids: list[int], end: str) -> str:
        """
        Let the model write a string value until it closes the quote.

        JSON escapes (like \\\\d) are decoded, and spaces left at the
        start or end by the tokenizer are removed.
        """
        token_id = self._next_token(ids, self.string_start)
        ids.append(token_id)
        first = self.texts[token_id]
        writen = first[first.index('"') + 1:]
        for _ in range(60):     # safety limit
            escape = self._ends_with_escape(writen)
            allowed = []
            for token_id, text in enumerate(self.texts):
                if text == "" or "\n" in text:
                    continue
                if '"' not in text:
                    allowed.append(token_id)
                    continue
                pos = self._first_unescaped_quote(text, escape)
                if pos == -1 or text[pos + 1:] in ("", end):
                    allowed.append(token_id)

            token_id = self._next_token(ids, allowed)
            text = self.texts[token_id]
            ids.append(token_id)
            pos = self._first_unescaped_quote(text, escape)
            if pos != -1:
                writen += text[:pos]
                if text[pos + 1:] == "":
                    self._add_manual_text(ids, end)
                break
            writen += text

        value: str
        try:
            value = json.loads('"' + writen + '"')
        except json.JSONDecodeError:
            value = writen
        return value.strip()

    def _read_json(self, path: str) -> Any:
        """Read a JSON file from the model (vocab.json)."""
        with Path(path).open('r', encoding='utf-8') as fd:
            return json.load(fd)

    def _validate_prompts(self, path: Path) -> list[dict[str, str]]:
        """Load and validate the prompts file."""
        prompts: list[dict[str, str]] = _PROMPT_ADAPTER.dump_python(
            load_prompt(path)
        )
        return prompts

    def _build_prompt(self, question: str, funcs: list[dict[str, Any]]) -> str:
        """Build the prompt in Qwen's tool-calling chat format."""
        tools = "\n".join(json.dumps(fn) for fn in funcs)

        prompt = (
            '<|im_start|>system\n'
            '# Tools\n\n'
            'You call exactly one function to assist with the user query.\n\n'
            'You are provided with function signatures within '
            '<tools></tools> XML tags:\n'
            '<tools>\n'
            f'{tools}\n'
            '</tools>\n\n'
            'For each function call, return a json object with function '
            'name and parameters within <tool_call></tool_call> XML tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "parameters": <args-json-object>}\n'
            '</tool_call><|im_end|>\n'
            '<|im_start|>user\n'
            f'{question}<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<think>\n\n</think>\n\n'
            '<tool_call>\n'
            '{"name": "'
        )

        return prompt

    def llm(
        self, question: str, funcs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate the function call for one question."""
        prompt = self._build_prompt(question, funcs)
        ids: list[int] = self.model.encode(prompt)[0].tolist()

        func_names = [str(f['name']) for f in funcs]
        func_name = self._chose_function(ids, func_names)

        func_def = next(f for f in funcs if f['name'] == func_name)

        params: dict[str, Any] = {}
        items = list(func_def['parameters'].items())
        for i, (param, type) in enumerate(items):
            end = "}" if i == len(items) - 1 else ","
            prefix = ', "parameters": {' if i == 0 else " "
            if type['type'] == 'string':
                self._add_manual_text(ids, prefix + '"' + param + '":')
            else:
                self._add_manual_text(ids, prefix + '"' + param + '": ')

            if type['type'] == 'number':
                params[param] = self._add_number(ids)
                self._add_manual_text(ids, end)
            elif type['type'] == 'integer':
                params[param] = self._add_number(ids, integer=True)
                self._add_manual_text(ids, end)
            elif type['type'] == 'string':
                params[param] = self._add_string(ids, end)
            elif type['type'] == "boolean":
                wrd = self._chose_function(ids, ["true", "false"], end="")
                params[param] = (wrd == 'true')
                self._add_manual_text(ids, end)

        return {"prompt": question, "name": func_name, "parameters": params}
