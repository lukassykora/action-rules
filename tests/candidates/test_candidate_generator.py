#!/usr/bin/env python
"""Bitset-focused tests for CandidateGenerator."""

import numpy as np
import pandas as pd
import pytest

from action_rules.action_rules import ActionRules
from action_rules.candidates.candidate_generator import CandidateGenerator
from action_rules.rules.rules import Rules


def _build_bit_masks(data: np.ndarray) -> np.ndarray:
    action_rules = ActionRules(
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_undesired_confidence=0.5,
        min_desired_support=1,
        min_desired_confidence=0.5,
    )
    action_rules.set_array_library(use_gpu=False, df=pd.DataFrame({'dummy': [0]}))
    return action_rules.build_bit_masks(data)


def _build_candidate_generator(
    data: np.ndarray,
    min_stable_attributes: int = 1,
    min_flexible_attributes: int = 1,
) -> CandidateGenerator:
    bit_masks = _build_bit_masks(data)
    rules = Rules(
        undesired_state='0',
        desired_state='1',
        columns=[f'c{i}' for i in range(data.shape[0])],
        count_transactions=data.shape[1],
    )
    return CandidateGenerator(
        min_stable_attributes=min_stable_attributes,
        min_flexible_attributes=min_flexible_attributes,
        min_undesired_support=1,
        min_desired_support=1,
        min_undesired_confidence=0.5,
        min_desired_confidence=0.5,
        undesired_state=0,
        desired_state=1,
        rules=rules,
        bit_masks=bit_masks,
        frames_bit_masks={0: bit_masks[0], 1: bit_masks[1]},
    )


@pytest.fixture
def sample_generator() -> CandidateGenerator:
    data = np.array(
        [
            [1, 1, 0, 0],  # target undesired
            [1, 0, 1, 0],  # target desired
            [1, 1, 1, 0],  # candidate item
        ],
        dtype=np.uint8,
    )
    return _build_candidate_generator(data, min_stable_attributes=1, min_flexible_attributes=0)


def test_init_bitset_fields(sample_generator):
    assert sample_generator.bit_masks is not None
    assert sample_generator.frames_bit_masks
    assert sample_generator.min_stable_attributes == 1


def test_get_support_bitset_matches_reference():
    data = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    candidate_generator = _build_candidate_generator(data)
    mask_vector = np.array([1, 0, 1, 0], dtype=np.uint8)
    mask_bitset = _build_bit_masks(mask_vector.reshape(1, -1))[0]

    item_index = 1
    expected_support = int(np.sum(data[item_index] * mask_vector))
    assert candidate_generator.get_support(mask_bitset, item_index) == expected_support


def test_bitset_support_batch_matches_reference():
    data = np.array(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    candidate_generator = _build_candidate_generator(data)
    mask_vector = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
    mask_bitset = _build_bit_masks(mask_vector.reshape(1, -1))[0]
    items = [0, 1, 2]

    expected_supports = [int(np.sum(data[item] * mask_vector)) for item in items]
    assert candidate_generator._bitset_support_batch(mask_bitset, items) == expected_supports


def test_generate_candidates_creates_bitset_branches(sample_generator):
    new_branches = sample_generator.generate_candidates(
        ar_prefix=tuple(),
        itemset_prefix=tuple(),
        stable_items_binding={'stable': [2]},
        flexible_items_binding={},
        actionable_attributes=0,
        stop_list=set(),
        stop_list_itemset=set(),
    )

    assert new_branches
    assert all('itemset_prefix' in branch for branch in new_branches)
    assert all('stable_items_binding' in branch for branch in new_branches)


def test_generate_candidates_batch_matches_single(sample_generator):
    candidate = {
        'ar_prefix': tuple(),
        'itemset_prefix': tuple(),
        'stable_items_binding': {'stable': [2]},
        'flexible_items_binding': {},
        'actionable_attributes': 0,
    }

    batched = sample_generator.generate_candidates_batch(
        [candidate],
        stop_list=set(),
        stop_list_itemset=set(),
        batch_size=8,
    )
    single = sample_generator.generate_candidates(
        **candidate,
        stop_list=set(),
        stop_list_itemset=set(),
    )

    assert len(batched) == len(single)
    assert [b['itemset_prefix'] for b in batched] == [s['itemset_prefix'] for s in single]


def test_update_new_branches(sample_generator):
    new_branches = [{'item': 2}]
    stable_candidates = {'stable': [2]}
    flexible_candidates = {'flex': [2]}
    sample_generator.update_new_branches(new_branches, stable_candidates, flexible_candidates)
    assert 'stable_items_binding' in new_branches[0]
    assert 'flexible_items_binding' in new_branches[0]


def test_in_stop_list(sample_generator):
    assert sample_generator.in_stop_list((1, 2), {(1, 2)}) is True
    assert sample_generator.in_stop_list((1, 3), {(1, 2)}) is False


