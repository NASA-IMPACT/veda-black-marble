#!/usr/bin/env python
"""
Finds potential pointless wrapper functions.

A wrapper is considered pointless if it:
1. Has minimal cyclomatic complexity
2. Only calls another function with minimal additional logic
3. Adds little to no value over calling the wrapped function directly

Detects patterns like:
- Single return statement calling another function
- Multiple statements but only: validate params → call function → return
- Functions that just add default parameters
- Functions that just rename another function
"""
import argparse
import ast
import csv
import pathlib
import sys
from typing import Optional


class WrapperAnalyzer(ast.NodeVisitor):
    """Analyzes functions to detect wrapper patterns."""
    
    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.current_function = None
        self.statements = []
        self.calls = []
        self.returns = []
        self.assignments = []
        self.conditions = []
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Don't recurse into nested functions."""
        if self.current_function is None:
            self.current_function = node
            self.generic_visit(node)
            
    def visit_Return(self, node: ast.Return):
        self.returns.append(node)
        if node.value and isinstance(node.value, ast.Call):
            self.calls.append(node.value)
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        self.calls.append(node)
        self.generic_visit(node)
        
    def visit_Assign(self, node: ast.Assign):
        self.assignments.append(node)
        self.generic_visit(node)
        
    def visit_If(self, node: ast.If):
        self.conditions.append(node)
        self.generic_visit(node)
        
    def visit_Assert(self, node: ast.Assert):
        # Assertions are often just parameter validation
        self.generic_visit(node)
        
    def analyze(self, node: ast.FunctionDef) -> Optional[dict]:
        """Analyze a function to determine if it's a pointless wrapper."""
        self.reset()
        self.visit(node)
        
        # Skip dunder methods and properties
        if node.name.startswith("__") or any(
            isinstance(dec, ast.Name) and dec.id == "property" 
            for dec in node.decorator_list
        ):
            return None
            
        # Calculate LOC (excluding docstring)
        start_line = node.lineno
        end_line = node.end_lineno
        loc = end_line - start_line + 1
        
        # Get the main logic (excluding docstring if present)
        body = node.body
        if body and isinstance(body[0], ast.Expr):
            # Handle both ast.Str (Python < 3.8) and ast.Constant (Python >= 3.8)
            expr_value = body[0].value
            if isinstance(expr_value, ast.Str):
                body = body[1:]  # Skip docstring
            elif isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
                body = body[1:]  # Skip docstring
            
        # Pattern 1: Single return statement
        if len(body) == 1 and isinstance(body[0], ast.Return):
            return_stmt = body[0]
            if return_stmt.value and isinstance(return_stmt.value, ast.Call):
                return self._analyze_single_return(node, return_stmt.value, loc)
                
        # Pattern 2: Multiple statements but trivial logic
        if len(body) <= 5:  # Reasonable threshold for simple wrappers
            # Check if it's primarily validation + call + return
            non_validation_stmts = []
            for stmt in body:
                if not self._is_validation_statement(stmt):
                    non_validation_stmts.append(stmt)
                    
            # If after removing validation, we just have assignment(s) + return
            if len(non_validation_stmts) <= 2:
                # Look for pattern: result = func(); return result
                if (len(self.returns) == 1 and self.returns[0].value and 
                    isinstance(self.returns[0].value, (ast.Call, ast.Name))):
                    
                    # Find the main call
                    main_call = None
                    if isinstance(self.returns[0].value, ast.Call):
                        main_call = self.returns[0].value
                    elif len(self.calls) == 1:
                        main_call = self.calls[0]
                        
                    if main_call:
                        return self._analyze_wrapper_call(node, main_call, loc, "validation_wrapper")
                        
        # Pattern 3: Just adds default parameters
        if len(body) == 1 and isinstance(body[0], ast.Return):
            return_stmt = body[0]
            if return_stmt.value and isinstance(return_stmt.value, ast.Call):
                call = return_stmt.value
                # Check if the call adds keyword arguments not in function signature
                func_params = {arg.arg for arg in node.args.args}
                call_keywords = {kw.arg for kw in call.keywords if kw.arg}
                if call_keywords - func_params:  # Has keywords not in params
                    return self._analyze_wrapper_call(node, call, loc, "default_params")
                    
        return None
        
    def reset(self):
        """Reset state for analyzing a new function."""
        self.current_function = None
        self.statements = []
        self.calls = []
        self.returns = []
        self.assignments = []
        self.conditions = []
        
    def _is_validation_statement(self, stmt) -> bool:
        """Check if a statement is likely parameter validation."""
        if isinstance(stmt, ast.Assert):
            return True
        if isinstance(stmt, ast.If):
            # Simple validation patterns like: if not x: raise
            if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Raise):
                return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            # Validation function calls like validate_params()
            call_name = self._get_call_name(stmt.value)
            if call_name and any(word in call_name.lower() for word in ["valid", "check", "assert"]):
                return True
        return False
        
    def _get_call_name(self, call: ast.Call) -> Optional[str]:
        """Extract the function name from a call."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        elif hasattr(ast, "unparse"):
            return ast.unparse(call.func)
        return None
        
    def _classify_wrapper_type(self, called_name: str) -> str:
        """Classify the wrapper based on what it calls."""
        called_lower = called_name.lower()
        
        if any(word in called_lower for word in ["read", "write", "open", "save", "load"]):
            return "io"
        elif any(word in called_lower for word in ["path", "file", "dir"]):
            return "path"
        elif "numpy" in called_lower or "np." in called_name:
            return "numpy"
        elif "rasterio" in called_lower or any(word in called_lower for word in ["raster", "tiff", "geotiff"]):
            return "rasterio"
        elif "gdal" in called_lower or "ogr" in called_lower:
            return "gdal"
        else:
            return "misc"
            
    def _analyze_single_return(self, node: ast.FunctionDef, call: ast.Call, loc: int) -> dict:
        """Analyze a single-return wrapper."""
        return self._analyze_wrapper_call(node, call, loc, "single_return")
        
    def _analyze_wrapper_call(self, node: ast.FunctionDef, call: ast.Call, loc: int, 
                              wrapper_pattern: str) -> dict:
        """Analyze a wrapper function call."""
        called_name = self._get_call_name(call) or "unknown"
        
        # Check parameter passthrough
        param_names = {arg.arg for arg in node.args.args}
        passed_params = 0
        
        for arg in call.args:
            if isinstance(arg, ast.Name) and arg.id in param_names:
                passed_params += 1
                
        for kw in call.keywords:
            if isinstance(kw.value, ast.Name) and kw.value.id in param_names:
                passed_params += 1
                
        passthrough_ratio = passed_params / len(param_names) if param_names else 0
        
        # Don't flag as wrapper if it's doing significant parameter transformation
        if passthrough_ratio < 0.5 and len(param_names) > 2:
            return None
            
        return {
            "is_wrapper": True,
            "function_name": node.name,
            "called_object": called_name,
            "wrapper_type": self._classify_wrapper_type(called_name),
            "wrapper_pattern": wrapper_pattern,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "loc": loc,
            "param_passthrough": passthrough_ratio,
        }


def find_wrappers_in_file(filepath: pathlib.Path, ignore_patterns: list[str], base_dir: pathlib.Path) -> list[dict]:
    """Find all potential wrapper functions in a file."""
    try:
        with open(filepath) as f:
            source = f.read()
            source_lines = source.splitlines()
            tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, OSError):
        return []
        
    analyzer = WrapperAnalyzer(source_lines)
    wrappers = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip if matches ignore pattern
            if any(pattern in node.name for pattern in ignore_patterns):
                continue
                
            analysis = analyzer.analyze(node)
            if analysis:
                # Calculate relative path from base directory
                try:
                    analysis["path"] = str(filepath.relative_to(base_dir))
                except ValueError:
                    # If not relative to base_dir, use absolute path
                    analysis["path"] = str(filepath)
                wrappers.append(analysis)
                
    return wrappers


def output_csv(wrappers: list[dict], output_file):
    """Output results in CSV format."""
    fieldnames = [
        "path", "function_name", "start_line", "end_line",
        "called_object", "wrapper_type", "LOC", "initial_status"
    ]
    
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    
    for wrapper in sorted(wrappers, key=lambda x: (x["path"], x["start_line"])):
        writer.writerow({
            "path": wrapper["path"],
            "function_name": wrapper["function_name"],
            "start_line": wrapper["start_line"],
            "end_line": wrapper["end_line"],
            "called_object": wrapper["called_object"],
            "wrapper_type": wrapper["wrapper_type"],
            "LOC": wrapper["loc"],
            "initial_status": "potential_wrapper"
        })


def output_text(wrappers: list[dict]):
    """Output results in text format."""
    print(f"Found {len(wrappers)} potential pointless wrappers:\n")
    
    # Group by wrapper type
    by_type = {}
    for wrapper in wrappers:
        wrapper_type = wrapper["wrapper_type"]
        if wrapper_type not in by_type:
            by_type[wrapper_type] = []
        by_type[wrapper_type].append(wrapper)
        
    for wrapper_type, typed_wrappers in sorted(by_type.items()):
        print(f"\n{wrapper_type.upper()} Wrappers ({len(typed_wrappers)}):")
        print("=" * 50)
        
        for wrapper in sorted(typed_wrappers, key=lambda x: (x["path"], x["start_line"])):
            print(f"\n{wrapper['path']}:{wrapper['start_line']}-{wrapper['end_line']} - {wrapper['function_name']}()")
            print(f"  Pattern: {wrapper['wrapper_pattern']}")
            print(f"  Calls: {wrapper['called_object']}")
            print(f"  LOC: {wrapper['loc']}")
            print(f"  Parameter passthrough: {wrapper['param_passthrough']:.0%}")


def main():
    """Find all pointless wrappers in the blackmarble package."""
    parser = argparse.ArgumentParser(description="Find pointless wrapper functions")
    parser.add_argument(
        "--format", 
        choices=["csv", "text"], 
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Ignore functions containing this pattern (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Default ignores
    ignore_patterns = ["__", "test_", "_test"] + args.ignore
    
    base_dir = pathlib.Path(__file__).parent.parent
    blackmarble_dir = base_dir / "blackmarble"
    
    all_wrappers = []
    
    # Search in both blackmarble package and acquire directory
    search_dirs = [blackmarble_dir]
    if (base_dir / "acquire").exists():
        search_dirs.append(base_dir / "acquire")
    
    for search_dir in search_dirs:
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                wrappers = find_wrappers_in_file(py_file, ignore_patterns, base_dir)
                all_wrappers.extend(wrappers)
            
    # Output results
    if args.output:
        output_file = open(args.output, "w", newline="")
    else:
        output_file = sys.stdout
        
    try:
        if args.format == "csv":
            output_csv(all_wrappers, output_file)
        else:
            output_text(all_wrappers)
    finally:
        if args.output:
            output_file.close()


if __name__ == "__main__":
    main()