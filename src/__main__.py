# from pathlib import Path

# from .json import load_funcs, load_prompt, write_output


# path = Path.cwd()/'data/input/function_calling_tests.json'
# path2 = Path.cwd()/'data/input/functions_definition.json'
# path3 = Path.cwd()/'data/output/function_calling_results.json'

# try:
#     print(load_prompt(path))
#     print()
#     print(load_funcs(path2))
#     print()
#     write_output(path3, '[{"prompt": "What is the sum of 2 and 3?","name": "fn_add_numbers","parameters": {"a": 2.0, "b": 3.0}}]')
# except Exception as e:
#     print(e)


# print("blablabla")

################################################
#                     LLM                      #
################################################






# print(repr(token_to_text("Ġthe")))   # ' the'
# print(repr(token_to_text("Ċ")))      # '\n'
# print(repr(token_to_text("ĠĠ")))     # '  '
# print(len(bytes_to_unicode()))       # 256


# print(bytes_to_unicode())


from src.llm import llm
from pathlib import Path
from src.deal_json import _FUNC_ADAPTER, load_funcs

path = Path.cwd()/'data/input/functions_definition.json'
funcs = _FUNC_ADAPTER.dump_python(load_funcs(path))

llm = llm()
questions = llm._validate_prompts()
# print(questions)
print(llm.llm(questions[0].get('prompt'), funcs))
# question = questions[0].get('prompt')
# prompt = llm._build_prompt(question, funcs)
# ids = llm.model.encode(prompt)[0].tolist()


# # print()
# # print(f"ids are {ids}")
# # print()
# # print(f"prompt is {prompt}")
# # model.llm(prompt)
# print()
# print(llm._chose_function(ids, [func.get('name') for func in funcs]))
# # for f in funcs:
# #     print(f.get('name'))
# # model._chose_function()
