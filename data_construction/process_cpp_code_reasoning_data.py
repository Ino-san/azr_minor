from pathlib import Path
import argparse
import re

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from azr_minor.data_construction.prompts import get_code_problem_predictor_prompt
from azr_minor.data_construction.process_data import instruction_following
from azr_minor.utils.code_utils.parsers import extract_function_from_cpp_string, parse_imports

def extract_input_from_cpp_string(cpp_code_content: str) -> str:
    # Extract the input from the C++ code content
    input_match = re.search(r'candidate\(\((.|\n)*?\=\=', cpp_code_content)
    if input_match:
        input = input_match.group(0)
        input = re.sub(r'candidate\(\(', '', input)
        input = re.sub(r'\)\) ==', '', input)
        return input
    else:
        raise ValueError("Could not find input in C++ code content") 
    
def extract_output_from_cpp_string(cpp_code_content: str) -> str:
    output_match = re.search(r'\=\=(.|\n)*', cpp_code_content)
    if output_match:
        output = output_match.group(0)
        output = re.sub(r'\=\= \(', '', output)
        output = re.sub(r'\);\n\}\n', '', output)
        return output
    else:
        raise ValueError("Could not find output in C++ code content")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--language', type=str, default="python",)
    args = parser.parse_args()

    # 283, 452, 510
    ds = load_dataset('xhwl/cruxeval-x')['Cpp']
    ds = ds.map(lambda x: {'problem': '\n'.join(parse_imports(x['code'], 'cpp')) + '\n' + extract_function_from_cpp_string(x['code']),
                            'input': extract_input_from_cpp_string(x['code']),
                            'output': extract_output_from_cpp_string(x['code'])
                           })
    output_data = []
    for i, data in enumerate(tqdm(ds, desc="Processing CruxEval-X")):
        prompt = get_code_problem_predictor_prompt(args.language, 'code_i', data['problem'], data['input'], data['output'])
        formatted_question = instruction_following.format(prompt)
        output_data.append({
            "data_source": 'cruxevalx_i',
            "prompt": [{
                "role": "user",
                "content": formatted_question
            }],
            "problem": data['problem'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['output']
            },
            "extra_info": {
                'split': 'test',
                'index': i,
                'metric': 'pred_code_i',
                'problem_type': 'code_i',
                'input': data['input'],
                'output': data['output'],
            }
        })
        prompt = get_code_problem_predictor_prompt(args.language, 'code_o', data['problem'], data['input'], data['output'])
        formatted_question = instruction_following.format(prompt)
        output_data.append({
            "data_source": 'cruxevalx_o',
            "prompt": [{
                "role": "user",
                "content": formatted_question
            }],
            "problem": data['problem'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['output']
            },
            "extra_info": {
                'split': 'test',
                'index': i + len(data),
                'metric': 'pred_code_o',
                'problem_type': 'code_o',
                'input': data['input'],
                'output': data['output'],
            }
        })


    df = pd.DataFrame(output_data)
    if args.max_length > 0:
        df = df.iloc[:args.max_length]
    path = Path('data/code_reason')
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f'cpp_test_answer{"_" + str(args.max_length) if args.max_length > 0 else ""}.parquet')
