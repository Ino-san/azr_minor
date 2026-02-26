from pathlib import Path
import argparse
import re

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from azr_minor.data_construction.prompts import get_code_problem_predictor_prompt
from azr_minor.data_construction.process_data import instruction_following

def extract_input_from_racket_string(racket_code_content: str) -> str:
    # Extract the input from the Rust code content
    input_match = re.search(r'check-within \(candidate(.|\n)*? \?\?\?\?', racket_code_content)
    if input_match:
        input = input_match.group(0)
        input = re.sub(r'check-within \(candidate ', '', input)
        input = re.sub(r'\) \?\?\?\?', '', input)
        return input
    else:
        raise ValueError("Could not find input in Rust code content")

def extract_output_from_racket_string(racket_code_content: str) -> str:
    output_match = re.search(r'check-within \(candidate \?\?\?\?\) (.|\n)*?\n', racket_code_content)
    if output_match:
        output = output_match.group(0)
        output = re.sub(r'check-within \(candidate \?\?\?\?\) ', '', output)
        output = re.sub(r' 0.001\)\n', '', output)
        return output
    else:
        raise ValueError("Could not find output in Rust code content")

def extract_function_from_racket_string(racket_code_content: str) -> str:
    extracted_lines = []
    in_function = False
    brace_level = 0
    # 関数のシグネチャを検出するための正規表現
    function_start_pattern = re.compile("\(define \(f ")
    # 文字列を改行で分割して行リストにする
    lines = racket_code_content.splitlines(keepends=True) # keepends=Trueで改行文字を保持
    for line in lines:
        extracted_lines.append(line)
        stripped_line = line.strip()
        if not in_function:
            # まだ関数定義に入っていない場合
            if function_start_pattern.search(stripped_line):
                current_line_open_braces = stripped_line.count('(')
                current_line_close_braces = stripped_line.count(')')
                brace_level += current_line_open_braces
                brace_level -= current_line_close_braces
                in_function = True
        else:
            # 関数本体内にある場合
            brace_level += stripped_line.count('(')
            brace_level -= stripped_line.count(')')
            if brace_level == 0:
                # ブレースレベルが0に戻ったら、関数の終わり
                in_function = False
                return "".join(extracted_lines) # 最初の関数定義が見つかったので終了
    return "" # 関数が見つからなかった場合


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--language', type=str, default="racket")
    args = parser.parse_args()

    # 283, 452, 510
    ds = load_dataset('xhwl/cruxeval-x')['Racket']
    ds = ds.map(lambda x: {'problem': extract_function_from_racket_string(x['code']),
                            'input': extract_input_from_racket_string(x['output_reasoning']),
                            'output': extract_output_from_racket_string(x['input_reasoning'])
                           })
    output_data = []
    print(ds[0]['input'])
    print(ds[0]['output'])
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
    df.to_parquet(path / f'racket_test_answer{"_" + str(args.max_length) if args.max_length > 0 else ""}.parquet')
