from pathlib import Path
import argparse
import re

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from azr_minor.data_construction.prompts import get_code_problem_predictor_prompt
from azr_minor.data_construction.process_data import instruction_following
from azr_minor.utils.code_utils.parsers import extract_function_from_cpp_string, parse_imports

def extract_input_from_go_string(go_code_content: str) -> str:
    # Extract the input from the Go code content
    input_match = re.search(r'actual: candidate(.|\n)*?, expected: ', go_code_content)
    if input_match:
        input = input_match.group(0)
        input = re.sub(r'actual: candidate\(', '', input)
        input = re.sub(r'\), expected: ', '', input)
        return input
    else:
        raise ValueError("Could not find input in Go code content")

def extract_output_from_go_string(go_code_content: str) -> str:
    output_match = re.search(r'expected: (.|\n)*?\},\n *?\}', go_code_content)
    if output_match:
        output = output_match.group(0)
        output = re.sub(r'expected\: ', '', output)
        output = re.sub(r' \},\n *?\}', '', output)
        return output
    else:
        raise ValueError("Could not find output in Go code content")

def extract_function_from_go_string(go_code_content: str) -> str:
    extracted_lines = []
    in_function = False
    in_import = False
    brace_level = 0
    go_code_content = re.sub('package f_test', 'package main', go_code_content)
    if 'package main' in go_code_content:
        # 関数のシグネチャを検出するための正規表現
        function_start_pattern = re.compile(r'\b' + "f" + r'\s*\(')
        import_pattern = re.compile(r'\b' + "import")

        # 文字列を改行で分割して行リストにする
        lines = go_code_content.splitlines(keepends=True) # keepends=Trueで改行文字を保持

        for line in lines:
            stripped_line = line.strip()
            if not in_import:
                extracted_lines.append(line)
                if import_pattern.search(stripped_line):
                    current_line_open_braces = stripped_line.count('(')
                    current_line_close_braces = stripped_line.count(')')
                    brace_level += current_line_open_braces
                    brace_level -= current_line_close_braces
                    in_import = True
            else:
                if not "testing" in stripped_line:
                    extracted_lines.append(line)
                brace_level += stripped_line.count('(')
                brace_level -= stripped_line.count(')')
                if brace_level == 0:
                    in_import = False
            if not in_function:
                # まだ関数定義に入っていない場合
                if function_start_pattern.search(stripped_line):
                    current_line_open_braces = stripped_line.count('{')
                    current_line_close_braces = stripped_line.count('}')
                    brace_level += current_line_open_braces
                    brace_level -= current_line_close_braces
                    in_function = True
            else:
                # 関数本体内にある場合
                brace_level += stripped_line.count('{')
                brace_level -= stripped_line.count('}')

                if brace_level == 0:
                    # ブレースレベルが0に戻ったら、関数の終わり
                    in_function = False
                    return "".join(extracted_lines) # 最初の関数定義が見つかったので終了
    return "" # 関数が見つからなかった場合


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--language', type=str, default="nodejs")
    args = parser.parse_args()

    # 283, 452, 510
    remove_index = set()
    ds = load_dataset('xhwl/cruxeval-x')['Go']
    ds = ds.map(lambda data: {
        'problem': extract_function_from_go_string(data['code']),
        'input': extract_input_from_go_string(data['output_reasoning']),
        'output': extract_output_from_go_string(data['input_reasoning'])
    })
    ds = ds.filter(lambda x: x['problem'] != "")
    output_data = []
    print(ds[0]['problem'])
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
    df.to_parquet(path / f'go_test_answer{"_" + str(args.max_length) if args.max_length > 0 else ""}.parquet')
