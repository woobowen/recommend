#!/usr/bin/env python3
"""Non-author static AST comparison; does not import Torch or execute a model."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
old_path = ROOT/'evidence/dependency_sources/peft_lora.py.txt'
new_path = ROOT/'scripts/compat.py'
old_tree = ast.parse(old_path.read_text())
old_class = next(n for n in ast.walk(old_tree) if isinstance(n, ast.ClassDef) and n.name == 'Linear8bitLt')
old = copy.deepcopy(next(n for n in old_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
new_tree = ast.parse(new_path.read_text())
new = copy.deepcopy(next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'peft05_out_of_place_forward'))

class ApprovedChange(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node.func) == 'super().forward':
            return ast.copy_location(ast.Call(func=ast.parse('bnb.nn.Linear8bitLt.forward', mode='eval').body,
                                             args=[ast.Name(id='self', ctx=ast.Load()), *node.args], keywords=node.keywords), node)
        return node

    def visit_AugAssign(self, node):
        if ast.unparse(node) == 'result += output':
            return ast.copy_location(ast.Assign(targets=[ast.Name(id='result', ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id='result', ctx=ast.Load()), op=ast.Add(), right=ast.Name(id='output', ctx=ast.Load()))), node)
        return self.generic_visit(node)

def strip_observation(statements):
    result = []
    for node in statements:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == 'evidence' for t in node.targets):
            continue
        if isinstance(node, ast.AugAssign) and ast.unparse(node.target).startswith('evidence['):
            continue
        if isinstance(node, ast.If):
            if ast.unparse(node.test) == 'evidence is None':
                continue
            node.body = strip_observation(node.body)
            node.orelse = strip_observation(node.orelse)
        result.append(node)
    return result

old = ApprovedChange().visit(old)
old.body = strip_observation(old.body)
new.body = strip_observation(new.body)
old.name = new.name = 'forward'
for function in [old, new]:
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
ast.fix_missing_locations(old)
ast.fix_missing_locations(new)
old_dump, new_dump = ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False)
assert old_dump == new_dump, 'Derived computational AST differs beyond explicitly approved changes'
print(json.dumps({'status':'PASS_STATIC_COMPUTATIONAL_AST', 'reviewer':'/root/independent_review',
    'original_file_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
    'compat_file_sha256':hashlib.sha256(new_path.read_bytes()).hexdigest(),
    'normalized_computational_ast_sha256':hashlib.sha256(new_dump.encode()).hexdigest(),
    'approved_changes':['Pinned explicit bnb parent call', 'Out-of-place residual addition'],
    'removed_for_comparison':['Docstring/imports', 'Module-call evidence assignment/guard/counters only'],
    'limits':'Static original-versus-derived path equivalence only. Does not establish CUDA numeric/Autograd correctness.'},indent=2))
