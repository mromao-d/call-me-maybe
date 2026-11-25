import json
from llm_sdk.llm_sdk import Small_LLM_Model
from pathlib import Path
from pydantic import BaseModel
import numpy as np
from .deal_json import load_prompt, load_funcs, _PROMPT_ADAPTER, _FUNC_ADAPTER
import re
from typing import Any


class llm:
    def __init__(self):
        self.model = Small_LLM_Model()

        self.vocab_path = self.model.get_path_to_vocab_file()
        self.tokenizer_path = self.model.get_path_to_tokenizer_file()
        self.vocab = self._read_json(self.vocab_path)
        self.byte_decoder = self._byte_decoder()
        self.tokenizer = self._read_json(self.tokenizer_path)
        self._id_to_token()

    def _bytes_to_unicode(self) -> dict[float: str]:
        """
        In resume: Transforms non printable chars into printable chars
        Retorna uma lista de bytes utf-8 e uma lista correspondente de strings unicode.
        Evita mapear para caracteres de espaço em branco/controlo que o código BPE não consegue processar.
        """
        # Intervalos de bytes que já são caracteres imprimíveis padrão
        printable: list[int] = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        mapping: dict[int, str] = {b: chr(b) for b in printable}
        extra = 0

        # Os restantes bytes (0-32, 127-160, 173) são deslocados para pontos Unicode acima de 256
        for b in range(256):
            if b not in mapping:
                mapping[b] = chr(256 + extra)
                extra += 1

        return mapping

    def _byte_decoder(self) -> dict[str, int]:
        return {char: byte for byte, char in self._bytes_to_unicode().items()}

    def _token_to_printable(self, token: str) -> str:
        """Turn all vocab.json tokens into the real values they stand for"""
        raw = bytes(self.byte_decoder[c] for c in token)
        return raw.decode("utf-8", errors="replace")

    def _id_to_token(self) -> None:
        self.vocab_size: int = len(self.model.get_logits_from_input_ids([0]))
        self.texts: list[str] = [""] * self.vocab_size
        for token, tid in self.vocab.items():
            if tid < self.vocab_size:
                self.texts[tid] = self._token_to_printable(token)

    def _get_end_tokens(self) -> tuple[int, int]:
        added_tokens = self.tokenizer.get('added_tokens')

        eof_token = next(
            (_.get('id') for _ in added_tokens if '<|endoftext|>' in _.get('content')),
            None
        )

        end_token = next(
            (_.get('id') for _ in added_tokens if '<|im_end|>' in _.get('content')),
            None
        )
        return eof_token, end_token

    def _next_token(self, ids: list[int], allowed: np.ndarray) -> int:
        """
        Allowed tokens. All output is tokens are in the input,
        so I can filter them in the beggining.
        Makes model faster
        """
        scores = np.asarray(self.model.get_logits_from_input_ids(ids))
        masked = np.full(self.vocab_size, -np.inf)
        masked[allowed] = scores[allowed]
        return int(np.argmax(masked))

    # def _chose_function(self, ids: list[int], names: list[str]) -> str:
    def _chose_function(
        self, ids: list[int], names: list[str], end: str = '"'
    ) -> str:
        """
        Chose the function name, one token at a time.
        Improves performance of model.
        """
        targets = [name + end for name in names]

        written = ""
        while written not in targets:
            allowed = []
            for token_id, text in enumerate(self.texts):
                if text == "":
                    continue
                # print()
                # print(f"writen is {written}")
                # print(f"text is {text}")
                # print()
                new = written + text
                for target in targets:
                    if target.startswith(new):
                        allowed.append(token_id)
                        break
            if len(allowed) == 0:
                raise ValueError("Could not choose a function")

            token_id = self._next_token(ids, np.array(allowed))
            ids.append(token_id)
            written = written + self.texts[token_id]

        return written[:len(written) - len(end)]


    def _add_manual_text(self, ids: list[int], text: str) -> None:
        """
        Append fixed text without asking the model.
        Improves performance (in JSON formating cases, etc)
        """
        ids.extend(self.model.encode(text)[0].tolist())

    def _is_number(self, text: str, full: bool = False) -> bool:
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

    def _add_number(self, ids: list[int]) -> float:
        """
        Write a number
        return the number as float
        """
        writen = ""
        for _ in range(20):     # can also define the limit on the init and use it everywhere
            allowed = []
            for token_id, text in enumerate(self.texts):
                if text == "":
                    continue
                if self._is_number(writen + text):
                    allowed.append(token_id)
                elif self._is_number(writen, True) and text[0] in ",}":
                    allowed.append(token_id)

            token_id = self._next_token(ids, np.array(allowed))
            text = self.texts[token_id]
            if text[0] in ",}":
                break
            ids.append(token_id)
            writen = writen + text

        if not self._is_number(writen, True):
            raise ValueError(f"Bad number: {writen!r}")
        return float(writen)

    def _ends_with_escape(self, text: str) -> bool:
        """True if text ends with an unfinished backslash escape."""
        count = 0
        for char in reversed(text):
            if char != "\\":
                break
            count += 1
        return count % 2 == 1

    def _add_string(self, ids: list[int]) -> str:
        writen = ""
        for _ in range(60):
            escape = self._ends_with_escape(writen)
            allowed = []
            for token_id, text in enumerate(self.texts):
                if text == "":
                    continue
                if text[0] == '"':
                    if not escape:
                        allowed.append(token_id)
                elif '"' not in text and "\n" not in text:
                    allowed.append(token_id)

            token_id = self._next_token(ids, np.array(allowed))
            text = self.texts[token_id]
            if text[0] == '"':
                break
            ids.append(token_id)
            writen += text

        try:
            value = json.loads('"' + writen + '"')
        except json.JSONDecodeError:
            value = writen
        return value.strip()

    def _read_json(self, path: str) -> json:
        with Path(path).open('r', encoding='utf-8') as fd:
            file: dict[str: int] = json.load(fd)
        return file

    def _validate_prompts(self, path: Path) -> dict[str, str]:
        # path = Path.cwd()/'data/input/function_calling_tests.json'
        prompts = _PROMPT_ADAPTER.dump_python(load_prompt(path))
        # print(prompts[0].get('prompt'))
        return prompts

    def _build_prompt(self, question: str, funcs) -> str:
        # prompts = self._validate_prompts()
        tools = "\n".join(json.dumps(fn) for fn in funcs)

        prompt = (
            '<|im_start|>system\n'
            '# Tools\n\n'
            'You call exactly one function to assist with the user query.\n\n'
            'You are provided with function signatures within <tools></tools> XML tags:\n'
            '<tools>\n'
            f'{tools}\n'
            '</tools>\n\n'
            'For each function call, return a json object with function name and parameters within <tool_call></tool_call> XML tags:\n'
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

    def llm(self, question: str, funcs: list[dict]) -> dict[str, Any]:
        # eof_token, end_token = self._get_end_tokens()
        prompt = self._build_prompt(question, funcs)
        ids = self.model.encode(prompt)[0].tolist()

        func_names = [f.get("name") for f in funcs]
        func_name = self._chose_function(ids, func_names)

        func_def = next((f for f in funcs if f.get('name') == func_name))

        params = {}
        first = True
        for param, type in func_def.get('parameters').items():
            if first:
                self._add_manual_text(ids, ', "parameters": {')
                first = False
            else:
                self._add_manual_text(ids, ", ")
            if type.get('type') == 'string':
                self._add_manual_text(ids, '"' + param + '": "')
            else:
                self._add_manual_text(ids, '"' + param + '": ')

            if type.get('type') == 'number':
                params[param] = self._add_number(ids)
            elif type.get('type') == 'string':
                params[param] = self._add_string(ids)
                self._add_manual_text(ids, '"')
            elif type.get('type') == "boolean":
                wrd = self._chose_function(ids, ["true", "false"], end="")
                params[param] = (wrd == 'true')

        return {"prompt": question, "name": func_name, "parameters": params}

