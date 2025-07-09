from typing import List


RUN_CODE_TEMPLATE = """{code}
repr(f({inputs}))"""

RUN_CODE_TEMPLATE_REPR = {
"python":"""{code}
print('<FINAL_REPR_SYMBOL>', repr(f({inputs})))""",
"javascript": """{code}
console.log('<FINAL_REPR_SYMBOL>', f({inputs}))""",
"java": """{code}
System.out.println("<FINAL_REPR_SYMBOL>" + f({inputs}));""",
"cpp": """{code}
std::cout << "<FINAL_REPR_SYMBOL>" << f({inputs}) << std::endl;""",
"go": """{code}
fmt.Println("<FINAL_REPR_SYMBOL>", f({inputs}))""",
}

VALIDATE_CODE_TEMPLATE = """{code}
repr(f({inputs}))"""

VALIDATE_CODE_TEMPLATE_REPR = {
"python":"""{code}
print('<FINAL_REPR_SYMBOL>', repr(f({inputs})))""",
"javascript": """{code}
console.log('<FINAL_REPR_SYMBOL>', f({inputs}))""",
"java": """{code}
System.out.println("<FINAL_REPR_SYMBOL>" + f({inputs}));""",
"cpp": """{code}
std::cout << "<FINAL_REPR_SYMBOL>" << f({inputs}) << std::endl;""",                                        
"go": """{code}
fmt.Println("<FINAL_REPR_SYMBOL>", f({inputs}))""",
}

EVAL_INPUT_PREDICTION_TEMPLATE = """{code}
{gold_output} == f({agent_input})"""

EVAL_INPUT_PREDICTION_TEMPLATE_REPR = {
"python":"""{code}
print('<FINAL_REPR_SYMBOL>', repr({gold_output} == f({agent_input})))""",
"javascript": """{code}
console.log('<FINAL_REPR_SYMBOL>', {gold_output} == f({agent_input}))""",
"java": """{code}
System.out.println("<FINAL_REPR_SYMBOL>" + ({gold_output} == f({agent_input})));""",
"cpp": """{code}
std::cout << "<FINAL_REPR_SYMBOL>" << ({gold_output} == f({agent_input})) << std::endl;""",
"go": """{code}
fmt.Println("<FINAL_REPR_SYMBOL>", {gold_output} == f({agent_input}))""",
}

EVAL_OUTPUT_PREDICTION_TEMPLATE = """{code}
eval({gold_output}) == eval({agent_output})"""

EVAL_OUTPUT_PREDICTION_TEMPLATE_REPR = {
"python": """{code}
print('<FINAL_REPR_SYMBOL>', repr(eval({gold_output}) == eval({agent_output})))""",
"javascript": """{code}
console.log('<FINAL_REPR_SYMBOL>', eval({gold_output}) == eval({agent_output}))""",
"java": """{code}
System.out.println("<FINAL_REPR_SYMBOL>" + (eval({gold_output}) == eval({agent_output})));""",
"cpp": """{code}
std::cout << "<FINAL_REPR_SYMBOL>" << (eval({gold_output}) == eval({agent_output})) << std::endl;""",
"go": """{code}
fmt.Println("<FINAL_REPR_SYMBOL>", eval({gold_output}) == eval({agent_output}))""",
}

CHECK_DETERMINISM_TEMPLATE = """{code}
returns = f({inputs})
if returns != f({inputs}):
    raise Exception('Non-deterministic code')
repr(returns)"""

CHECK_DETERMINISM_TEMPLATE_REPR = {
"python": """{code}
returns = f({inputs})
if returns != f({inputs}):
    raise Exception('Non-deterministic code')
print('<FINAL_REPR_SYMBOL>', repr(returns))""",
"javascript": """{code}
let returns = f({inputs});
if (returns !== f({inputs})) {
    throw new Error('Non-deterministic code');
}
console.log('<FINAL_REPR_SYMBOL>', returns)""",
"java": """{code}
Object returns = f({inputs});
if (!returns.equals(f({inputs}))) {
    throw new Exception('Non-deterministic code');
}
System.out.println("<FINAL_REPR_SYMBOL>" + returns);""",
"cpp": """{code}
auto returns = f({inputs});
if (returns != f({inputs})) {
    throw std::runtime_error("Non-deterministic code");
}
std::cout << "<FINAL_REPR_SYMBOL>" << returns << std::endl;""",
"go": """{code}
returns := f({inputs})
if returns != f({inputs}) {
    panic("Non-deterministic code")
}
fmt.Println("<FINAL_REPR_SYMBOL>", returns)""",
}

output_string = {
"python":"""{code}
acc_list = []""",
"javascript": """{code}
let acc_list = []""",   
"java": """{code}
List<Boolean> acc_list = new ArrayList<>();""",
"cpp": """{code}
std::vector<bool> acc_list;""",
"go": """{code}
acc_list := []bool{}""",
}

append_string = {
"python": """\ntry:
    acc_list.append({gold_output} == f({inp}))
except:
    acc_list.append(False)""",
"javascript": """\ntry {
    acc_list.push({gold_output} === f({inp}));
} catch {
    acc_list.push(false);
}""",
"java": """\ntry {
    acc_list.add({gold_output} == f({inp}));
} catch {
    acc_list.add(false);
}""",
"cpp": """\ntry {
    acc_list.push_back({gold_output} == f({inp}));
} catch (...) {
    acc_list.push_back(false);
}""",
"go": """\ntry {
    acc_list = append(acc_list, {gold_output} == f({inp}))
} catch {
    acc_list = append(acc_list, false)
}""",
    }

repr_string = {
"python": {True: """\nprint('<FINAL_REPR_SYMBOL>', repr(acc_list))""", False: """\nacc_list"""},
"javascript": {True: """\nconsole.log('<FINAL_REPR_SYMBOL>', acc_list)""", False: """\nacc_list"""},
"java": {True: """\nSystem.out.println("<FINAL_REPR_SYMBOL>" + acc_list);""", False: """\nacc_list"""},
"cpp": {True: """\nstd::cout << "<FINAL_REPR_SYMBOL>" << acc_list << std::endl;""", False: """\nacc_list"""},
"go": {True: """\nfmt.Println("<FINAL_REPR_SYMBOL>", acc_list)""", False: """\nacc_list"""},
}

def EVAL_K_INPUT_PREDICTION_TEMPLATE(language: str, code: str, gold_output: str, k_agent_inputs: List[str], repr_output: bool = False):
    out_str = output_string[language].format(code=code)
    for inp in k_agent_inputs:
        out_str += append_string[language].format(gold_output=gold_output, inp=inp)
    out_str += repr_string[language][repr_output]
    return out_str

def EVAL_K_OUTPUT_PREDICTION_TEMPLATE(language: str, code: str, gold_output: str, k_agent_outputs: List[str], repr_output: bool = False):
    out_str = output_string[language].format(code=code)
    for out in k_agent_outputs:
        out_str += append_string[language].format(gold_output=gold_output, inp=out)
    out_str += repr_string[language][repr_output]
    return out_str
