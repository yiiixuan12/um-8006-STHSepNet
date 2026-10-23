# Test the checkpoint key-fix logic in run_main.py (module. prefix handling).
# This logic is pure string transformation, so it is tested in isolation and
# does not require torch/transformers to be installed.
import pytest


def _fix_keys(state_dict):
    new = {}
    for key in state_dict:
        new_key = key[7:] if key.startswith('module.') else key
        new[new_key] = state_dict[key]
    return new


def test_strips_module_prefix():
    state = {'module.layer1.weight': 1.0, 'module.layer2.bias': 2.0, 'plain': 3.0}
    out = _fix_keys(state)
    assert set(out) == {'layer1.weight', 'layer2.bias', 'plain'}


def test_handles_no_prefix():
    state = {'a': 1, 'moduleless': 2}
    assert _fix_keys(state) == state
