#!/usr/bin/env python
"""
Finds potential pointless wrapper functions.
A wrapper is considered pointless if it:
1. Has minimal cyclomatic complexity
2. Contains a single return statement
3. The return statement is a direct function call
"""
import ast
import pathlib


def analyze_function(node: ast.FunctionDef) -> dict:
    """Analyze a function to determine if it's a pointless wrapper."""
    # Count statements in the function body
    stmt_count = len(node.body)

    # Check if it's a single return statement
    if stmt_count == 1 and isinstance(node.body[0], ast.Return):
        return_stmt = node.body[0]
        if return_stmt.value and isinstance(return_stmt.value, ast.Call):
            # Check if it's calling another function with same/similar args
            call = return_stmt.value
            func_name = ast.unparse(call.func) if hasattr(ast, "unparse") else str(call.func)

            # Count parameters passed through
            param_names = {arg.arg for arg in node.args.args}
            passed_params = 0

            for arg in call.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    passed_params += 1

            return {
                "is_wrapper": True,
                "function_name": node.name,
                "calls": func_name,
                "param_passthrough": passed_params / len(param_names) if param_names else 0,
                "line": node.lineno,
            }

    return {"is_wrapper": False}


def find_wrappers_in_file(filepath: pathlib.Path) -> list[dict]:
    """Find all potential wrapper functions in a file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except (SyntaxError, OSError):
        return []

    wrappers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            analysis = analyze_function(node)
            if analysis["is_wrapper"]:
                analysis["file"] = str(filepath)
                wrappers.append(analysis)

    return wrappers


def main():
    """Find all pointless wrappers in the blackmarble package."""
    base_dir = pathlib.Path(__file__).parent.parent
    blackmarble_dir = base_dir / "blackmarble"

    all_wrappers = []

    for py_file in blackmarble_dir.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            wrappers = find_wrappers_in_file(py_file)
            all_wrappers.extend(wrappers)

    # Print results
    print(f"Found {len(all_wrappers)} potential pointless wrappers:\n")

    for wrapper in sorted(all_wrappers, key=lambda x: x["file"]):
        print(f"{wrapper['file']}:{wrapper['line']} - {wrapper['function_name']}()")
        print(f"  Calls: {wrapper['calls']}")
        print(f"  Parameter passthrough: {wrapper['param_passthrough']:.0%}")
        print()


if __name__ == "__main__":
    main()
