"""Function calling with Qwen3-0.6B using constrained decoding."""
import json
from llm_sdk import Small_LLM_Model
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import numpy as np
from .deal_json import load_prompt, _PROMPT_ADAPTER
from typing import Any


class llm(BaseModel):
    """Generate function calls with Qwen3-0.6B and constrained decoding.

    Loads the model once, decodes the whole vocabulary into real text, and
    then produces one function call per question.

    Attributes:
        model: The loaded Small_LLM_Model.
        vocab: Maps each vocab.json token string to its token id.
        byte_decoder: Maps each vocab.json character back to its byte.
        vocab_size: Number of logits the model returns.
        texts: Real text of each token, indexed by token id.
        string_start: Ids of the tokens that can open a JSON string.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = None
    vocab: dict[str, int] = {}
    byte_decoder: dict[str, int] = {}
    vocab_size: int = 0
    texts: list[str] = []
    string_start: list[int] = []

    def model_post_init(self, context: Any) -> None:
        """Load the model and prepare the token tables.

        Args:
            context: Pydantic validation context (unused).
        """
        self.model = Small_LLM_Model()

        vocab_path = self.model.get_path_to_vocab_file()
        self.vocab = self._read_json(vocab_path)
        self.byte_decoder = self._byte_decoder()
        self._id_to_token()
        self.string_start = self._string_start_tokens()

    def _string_start_tokens(self) -> list[int]:
        """Find the tokens that can open a JSON string value.

        A token qualifies if it starts with '"' or ' "' and the rest is
        string content: no newline and no unescaped quote.

        Returns:
            The ids of the tokens that can start a string value.
        """
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
        """Map every byte (0-255) to its character in vocab.json.

        Bytes that are already printable map to themselves. The others
        (spaces, control characters) are moved to characters above 256,
        for example a space becomes 'Ġ' and a newline becomes 'Ċ'.

        Returns:
            A mapping from each byte value to its stand-in character.
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
        """Build the reverse of _bytes_to_unicode.

        Returns:
            A mapping from each vocab.json character to its original byte.
        """
        return {char: byte for byte, char in self._bytes_to_unicode().items()}

    def _token_to_printable(self, token: str) -> str:
        """Convert a vocab.json token into the real text it stands for.

        Args:
            token: The token as written in vocab.json, e.g. 'Ġhello'.

        Returns:
            The decoded text, e.g. ' hello'. Incomplete UTF-8 sequences are
            replaced with U+FFFD.
        """
        raw = bytes(self.byte_decoder[c] for c in token)
        return raw.decode("utf-8", errors="replace")

    def _id_to_token(self) -> None:
        """Fill self.texts so that texts[id] is the real text of token id.

        Also sets self.vocab_size from the number of logits the model
        returns. Ids without a vocab.json entry keep an empty string.
        """
        self.vocab_size = len(self.model.get_logits_from_input_ids([0]))
        self.texts = [""] * self.vocab_size
        for token, tid in self.vocab.items():
            if tid < self.vocab_size:
                self.texts[tid] = self._token_to_printable(token)

    def _next_token(self, ids: list[int], allowed: list[int]) -> int:
        """Ask the model for the next token, among the allowed ones only.

        Every token that is not allowed gets a score of -inf, so argmax
        can only pick an allowed token.

        Args:
            ids: Token ids of the text so far.
            allowed: Ids of the tokens that may be chosen.

        Returns:
            The id of the allowed token with the highest score.
        """
        scores = np.asarray(self.model.get_logits_from_input_ids(ids))
        masked = np.full(self.vocab_size, -np.inf)
        masked[allowed] = scores[allowed]
        return int(np.argmax(masked))

    def _chose_function(
        self, ids: list[int], names: list[str], end: str = '"'
    ) -> str:
        """Let the model write one of the given names, one token at a time.

        A token is allowed only if the text written so far stays the start
        of some name followed by end. Chosen tokens are appended to ids.

        Args:
            ids: Token ids of the text so far. Modified in place.
            names: The options the model can choose from.
            end: Text that must follow the name, e.g. the closing quote.

        Returns:
            The chosen name, without end.

        Raises:
            ValueError: If no token can continue any of the names.
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
        """Append fixed text to the ids without asking the model.

        This saves model calls and guarantees the JSON structure.

        Args:
            ids: Token ids of the text so far. Modified in place.
            text: The text to encode and append.
        """
        ids.extend(self.model.encode(text)[0].tolist())

    def _is_number(
        self, text: str, full: bool = False, integer: bool = False
    ) -> bool:
        """Check whether text is a valid number, or the start of one.

        Args:
            text: The text to check.
            full: If False, accept text that can still become a number
                ('-', '12', '3.'). If True, accept only a complete number
                ('12', '-3.5').
            integer: If True, reject any decimal point.

        Returns:
            True if the text passes the check.
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
        """Let the model write a number value.

        Generation stops when the model picks a token starting with ',' or
        '}' after a complete number. That token is not kept in ids.

        Args:
            ids: Token ids of the text so far. Modified in place.
            integer: If True, the number cannot contain a decimal point.

        Returns:
            The number, as an int if integer is True, otherwise as a float.

        Raises:
            ValueError: If the generated text is not a complete number.
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
        """Check whether text ends with an unfinished backslash escape.

        Args:
            text: The string content written so far.

        Returns:
            True if text ends with an odd number of backslashes.
        """
        count = 0
        for char in reversed(text):
            if char != "\\":
                break
            count += 1
        return count % 2 == 1

    def _first_unescaped_quote(self, text: str, escape: bool) -> int:
        """Find the first double quote in text that is not escaped.

        Args:
            text: The text of one token.
            escape: True if the text before this token ends with an
                unfinished backslash, so the first character is escaped.

        Returns:
            The index of the first unescaped '"', or -1 if there is none.
        """
        for i, char in enumerate(text):
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                return i
        return -1

    def _add_string(self, ids: list[int], end: str) -> str:
        """Let the model write a string value, including both quotes.

        The first token comes from string_start, so the model writes the
        opening quote itself. After that, a token is allowed if it is
        content, or content followed by the closing quote and optionally
        end. The closing token is kept in ids; if it does not contain end,
        end is appended manually. At most 60 tokens follow the opening one.

        Args:
            ids: Token ids of the text so far. Modified in place.
            end: The character after the value: ',' or '}'.

        Returns:
            The value with JSON escapes decoded and surrounding spaces
            removed.
        """
        token_id = self._next_token(ids, self.string_start)
        ids.append(token_id)
        first = self.texts[token_id]
        writen = first[first.index('"') + 1:]
        for _ in range(60):
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
        """Read a JSON file, such as the model's vocab.json.

        Args:
            path: Path to the JSON file.

        Returns:
            The parsed JSON content.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with Path(path).open('r', encoding='utf-8') as fd:
            return json.load(fd)

    def _validate_prompts(self, path: Path) -> list[dict[str, str]]:
        """Load and validate the prompts file.

        Args:
            path: Path to the prompts JSON file.

        Returns:
            One dict per prompt, each with a 'prompt' key.

        Raises:
            ReadJsonException: If the file cannot be read or is invalid.
        """
        prompts: list[dict[str, str]] = _PROMPT_ADAPTER.dump_python(
            load_prompt(path)
        )
        return prompts

    def _build_prompt(self, question: str, funcs: list[dict[str, Any]]) -> str:
        """Build the prompt in Qwen's tool-calling chat format.

        The text ends right where the function name starts, so the next
        token the model writes is part of the name.

        Args:
            question: The user's natural-language request.
            funcs: The function definitions, as dicts.

        Returns:
            The full prompt text.
        """
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
        """Generate the function call for one question.

        The model chooses the function name, then writes a value for each
        parameter. The JSON structure around the values is added manually.

        Args:
            question: The user's natural-language request.
            funcs: The available function definitions, as dicts.

        Returns:
            A dict with the keys 'prompt', 'name' and 'parameters'.

        Raises:
            ValueError: If no function name can be generated, or a number
                is invalid.
        """
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
