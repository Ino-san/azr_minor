import ast
import re
from typing import List

import_string = {
    "python": "from|import",
    "js": "import",
    "java": "import",
    "cpp": "#include",
    "go": "import",
}

def parse_imports(code_snippet: str, language: str) -> List[str]:
    imports = []
    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Reconstruct import line from AST node
                if isinstance(node, ast.Import):
                    import_line = "import " + ", ".join(
                        [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    )
                else:
                    module = node.module or ""
                    import_line = f"from {module} import " + ", ".join(
                        [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    )
                    if node.level > 0:
                        import_line = f"from {'.' * node.level}{module} import " + ", ".join(
                            [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                                for alias in node.names]
                        )
                imports.append(import_line)
    except Exception as e:
        import_pattern = rf"^\s*(?:{import_string[language]})\s+.*$"
        imports = [i.strip() for i in re.findall(import_pattern, code_snippet, re.MULTILINE)]
    return imports


def parse_error(error_message: str) -> str:
    # split by colon
    error_message = error_message.split(':')[0]
    return error_message.strip()


def replace_main_function_name(code: str, old_name: str, new_name: str) -> str:
    """
    Replace all occurrences of `old_name` with `new_name` in the code.
    Replace the definition and all recursive calls of `old_name` with `new_name`.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == old_name:
            node.func.id = new_name
    return ast.unparse(tree)


def remove_comments_and_docstrings(code: str) -> str:
    """
    Remove all comments and docstrings from the code.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
                # Remove all leading docstrings
                while node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0].value
                    if isinstance(expr, (ast.Str, ast.Constant)) and (
                        isinstance(expr.value, str) if isinstance(expr, ast.Constant) else True
                    ):
                        node.body.pop(0)
                    else:
                        break
        
        # Convert back to code - AST unparse already removes comments
        code_without_docstrings = ast.unparse(tree)
        
        # Only remove empty lines and trim whitespace
        lines = [
            line.rstrip()
            for line in code_without_docstrings.split('\n')
            if line.strip()
        ]
        
        return '\n'.join(lines)
    except Exception as e:
        return code  # Return original code if parsing fails


def remove_any_not_definition_imports(code: str) -> str:
    """
    Remove anything that is not a definition or import.
    Preserves: 
    - Import/From imports
    - Class definitions
    - Function/AsyncFunction definitions
    Removes:
    - Top-level assignments
    - Standalone expressions
    - Constant declarations
    """
    class DefinitionFilter(ast.NodeTransformer):
        def visit_Module(self, node):
            # Keep only definitions and imports (explicitly exclude assignments)
            node.body = [
                n for n in node.body
                if isinstance(n, (
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef
                ))
            ]
            return node

    try:
        tree = ast.parse(code)
        tree = DefinitionFilter().visit(tree)
        ast.fix_missing_locations(tree)

        # Remove empty lines and format
        cleaned = ast.unparse(tree)
        return '\n'.join([line for line in cleaned.split('\n') if line.strip()])

    except Exception as e:
        return code


class PrintRemover(ast.NodeTransformer):
    def visit_Expr(self, node):
        # Handle top-level print statements
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return None
        return node

    def visit_Call(self, node):
        # Handle print calls in other contexts (like assignments)
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            return ast.Constant(value=None)
        return node

    def _handle_block(self, node):
        self.generic_visit(node)
        if not node.body:
            node.body.append(ast.Pass())
        return node

    def visit_For(self, node):
        return self._handle_block(node)

    def visit_While(self, node):
        return self._handle_block(node)

    def visit_FunctionDef(self, node):
        return self._handle_block(node)

    def visit_AsyncFunctionDef(self, node):
        return self._handle_block(node)

    def visit_If(self, node):
        return self._handle_block(node)

    def visit_With(self, node):
        return self._handle_block(node)

    def visit_Try(self, node):
        self.generic_visit(node)
        
        # Handle main try body
        if not node.body:
            node.body.append(ast.Pass())
            
        # Handle except handlers
        for handler in node.handlers:
            if not handler.body:
                handler.body.append(ast.Pass())
                
        # Handle else clause
        if node.orelse and not node.orelse:
            node.orelse.append(ast.Pass())
            
        # Handle finally clause
        if node.finalbody and not node.finalbody:
            node.finalbody.append(ast.Pass())
            
        return node


def remove_print_statements(code: str) -> str:
    """
    Remove all print statements from the code.
    """
    tree = ast.parse(code)
    tree = PrintRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def extract_function_from_cpp_string(cpp_code_content: str) -> str:
    extracted_lines = []
    in_function = False
    brace_level = 0
    function_signature_line_buffer = []

    # 関数のシグネチャを検出するための正規表現
    function_start_pattern = re.compile(r'\b' + "f" + r'\s*\(')

    # 文字列を改行で分割して行リストにする
    lines = cpp_code_content.splitlines(keepends=True) # keepends=Trueで改行文字を保持

    for line in lines:
        stripped_line = line.strip()

        if not in_function:
            # まだ関数定義に入っていない場合
            if function_start_pattern.search(stripped_line):
                # 関数シグネチャの開始を検出
                function_signature_line_buffer = [line]
                
                current_line_open_braces = stripped_line.count('{')
                current_line_close_braces = stripped_line.count('}')
                brace_level += current_line_open_braces
                brace_level -= current_line_close_braces

                if brace_level > 0: # もしこの行でブレースが開いていれば、関数本体に入ったとみなす
                    extracted_lines.extend(function_signature_line_buffer)
                    in_function = True
                    function_signature_line_buffer = []
            elif function_signature_line_buffer:
                # シグネチャが複数行にわたる可能性があるため、バッファリングを続ける
                function_signature_line_buffer.append(line)
                
                current_line_open_braces = stripped_line.count('{')
                current_line_close_braces = stripped_line.count('}')
                brace_level += current_line_open_braces
                brace_level -= current_line_close_braces

                if brace_level > 0: # 最初の '{' が見つかった場合
                    extracted_lines.extend(function_signature_line_buffer)
                    in_function = True
                    function_signature_line_buffer = []
                elif brace_level < 0: # シグネチャ行に閉じブレースがあったが開きブレースがなかった場合（エラーケース）
                    function_signature_line_buffer = []
                    brace_level = 0 # リセット
        else:
            # 関数本体内にある場合
            extracted_lines.append(line)
            brace_level += stripped_line.count('{')
            brace_level -= stripped_line.count('}')

            if brace_level == 0:
                # ブレースレベルが0に戻ったら、関数の終わり
                in_function = False
                return "".join(extracted_lines) # 最初の関数定義が見つかったので終了
    return "" # 関数が見つからなかった場合


def extract_function_from_js_string(js_code_content: str) -> str:
    extracted_lines = []
    in_function = False
    brace_level = 0
    function_signature_line_buffer = []

    # 関数のシグネチャを検出するための正規表現
    function_start_pattern = re.compile(r'\b' + "f" + r'\s*\(')

    # 文字列を改行で分割して行リストにする
    lines = cpp_code_content.splitlines(keepends=True) # keepends=Trueで改行文字を保持

    for line in lines:
        stripped_line = line.strip()

        if not in_function:
            # まだ関数定義に入っていない場合
            if function_start_pattern.search(stripped_line):
                # 関数シグネチャの開始を検出
                function_signature_line_buffer = [line]
                
                current_line_open_braces = stripped_line.count('{')
                current_line_close_braces = stripped_line.count('}')
                brace_level += current_line_open_braces
                brace_level -= current_line_close_braces

                if brace_level > 0: # もしこの行でブレースが開いていれば、関数本体に入ったとみなす
                    extracted_lines.extend(function_signature_line_buffer)
                    in_function = True
                    function_signature_line_buffer = []
            elif function_signature_line_buffer:
                # シグネチャが複数行にわたる可能性があるため、バッファリングを続ける
                function_signature_line_buffer.append(line)
                
                current_line_open_braces = stripped_line.count('{')
                current_line_close_braces = stripped_line.count('}')
                brace_level += current_line_open_braces
                brace_level -= current_line_close_braces

                if brace_level > 0: # 最初の '{' が見つかった場合
                    extracted_lines.extend(function_signature_line_buffer)
                    in_function = True
                    function_signature_line_buffer = []
                elif brace_level < 0: # シグネチャ行に閉じブレースがあったが開きブレースがなかった場合（エラーケース）
                    function_signature_line_buffer = []
                    brace_level = 0 # リセット
        else:
            # 関数本体内にある場合
            extracted_lines.append(line)
            brace_level += stripped_line.count('{')
            brace_level -= stripped_line.count('}')

            if brace_level == 0:
                # ブレースレベルが0に戻ったら、関数の終わり
                in_function = False
                return "".join(extracted_lines) # 最初の関数定義が見つかったので終了
    return "" # 関数が見つからなかった場合


if __name__ == "__main__":
    print(parse_error("NameError: name 'x' is not defined"))
    print(parse_error("TypeError: unsupported operand type(s) for -: 'str' and 'str'"))
    print(parse_error("ValueError: invalid literal for int() with base 10: 'x'"))
