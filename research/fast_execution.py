"""Execution-only changes; leave the frozen model/trainer sources untouched.

Keep every forward/backward, finite check, reduction, optimizer update and RNG
draw in the same order. Batch the *logging* transfers without moving a finite
check, and cache only immutable candidate addresses / group labels, never data.
AST edits fail closed if the frozen implementation no longer matches.
"""
from __future__ import annotations
import ast
import inspect
import textwrap
import types
import torch
from models.periodic_bert_fast import FastPeriodicBert
from models.periodic_fast_hourly import HourlyFastPeriodicBert


def _compile(function, tree, extra):
    namespace = dict(function.__globals__)
    namespace.update(extra)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), 'exec'), namespace)
    return namespace[function.__name__]


def deferred_utility_statistic(function):
    """Do not fetch the unused abstention-rate statistic from the GPU."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    matches = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            value = node.value.elts[1]
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'float':
                if len(value.args) != 1:
                    raise ValueError('utility statistic signature changed')
                node.value.elts[1] = value.args[0]
                matches += 1
    if matches != 1:
        raise ValueError('expected exactly one unused float utility statistic')
    return _compile(function, tree, {})


def objective_from_chunks(chunks):
    # One transfer; preserve the old Python-double summation / weighting order.
    values = torch.stack([value for value, _, _ in chunks]).cpu().tolist()
    objective = 0.
    for value, (_, chunk_size, node_count) in zip(values, chunks):
        objective += value * chunk_size / node_count
    return objective


def training_function(function, utility_loss):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))

    class ChangeLogging(ast.NodeTransformer):
        initializations = 0
        appends = 0
        resolutions = 0

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'objective' and isinstance(node.value, ast.Constant)
                    and node.value.value == 0.):
                self.initializations += 1
                return [node, ast.parse('_execution_chunks = []').body[0]]
            return node

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'objective':
                value = node.value
                if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
                        and isinstance(value.left, ast.BinOp) and isinstance(value.left.op, ast.Mult)
                        and ast.unparse(value.left.left) == 'float(loss.detach())'):
                    raise ValueError('objective update changed')
                self.appends += 1
                return ast.Expr(ast.Call(ast.Attribute(ast.Name('_execution_chunks', ast.Load()), 'append', ast.Load()),
                    [ast.Tuple([ast.parse('loss.detach()', mode='eval').body,
                        value.left.right, value.right], ast.Load())], []))
            return node

        def visit_For(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == 'start':
                if 'args.target_chunk' not in ast.unparse(node.iter):
                    raise ValueError('chunk loop changed')
                self.resolutions += 1
                return [node, ast.parse('objective = _execution_objective(_execution_chunks)').body[0]]
            return node

    change = ChangeLogging()
    tree = change.visit(tree)
    if (change.initializations, change.appends, change.resolutions) != (1, 1, 1):
        raise ValueError('frozen training loop no longer matches execution edit')
    return _compile(function, tree, {'_execution_objective': objective_from_chunks,
        'utility_training_loss': deferred_utility_statistic(utility_loss)})


def cached_candidates(original):
    cache = {}

    def candidates(history):
        # Invalid shapes still go through the original validation.
        key = (history.device, tuple(history.shape[1:]))
        if history.ndim != 2 or key not in cache:
            values, valid, indices = original(history)
            cache[key] = (valid, indices, indices.clamp(0, history.shape[1] - 1))
            return values, valid, indices
        valid, indices, clipped = cache[key]
        return history.float()[:, clipped].flatten(1, 2), valid, indices

    return candidates


_GROUP_CACHE = {}


def fixed_groups(device):
    device = torch.device(device)
    if device not in _GROUP_CACHE:
        # Exactly the labels made by repeat_interleave in the frozen model;
        # no CUDA tensor-repeats length discovery / synchronization is needed.
        labels = [0] * 16 + [1] * 8
        for group in range(2, 10):
            labels.extend([group] * 4)
        _GROUP_CACHE[device] = torch.tensor(labels, device=device, dtype=torch.long)
    return _GROUP_CACHE[device]


def execution_chunk():
    function = FastPeriodicBert._chunk
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'groups' for x in node.targets):
            if not ast.unparse(node.value).startswith('torch.repeat_interleave('):
                raise ValueError('group-label construction changed')
            node.value = ast.parse('_execution_groups(history.device)', mode='eval').body
            changes += 1
    if changes != 1:
        raise ValueError('expected one fixed group-label expression')
    return _compile(function, tree, {'_execution_groups': fixed_groups})


_CHUNK = execution_chunk()


def install_model_execution(model):
    model._chunk = types.MethodType(_CHUNK, model)
    model.pattern_operator.candidates = cached_candidates(model.pattern_operator.candidates)
    return model


def create_fast(*args, **kwargs):
    return install_model_execution(FastPeriodicBert(*args, **kwargs))


def create_hourly(*args, **kwargs):
    return install_model_execution(HourlyFastPeriodicBert(*args, **kwargs))


def install_training(shared):
    original = shared.train_stage

    def train_stage(base, student, datasets, args, report, output, epochs):
        if student is None:
            return original(base, student, datasets, args, report, output, epochs)
        # Read the currently injected utility helper: hourly and 5-minute
        # datasets keep their distinct, already checked causal teachers.
        function = training_function(original, shared.utility_training_loss)
        return function(base, student, datasets, args, report, output, epochs)

    shared.train_stage = train_stage
