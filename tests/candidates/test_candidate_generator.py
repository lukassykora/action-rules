#!/usr/bin/env python
"""Tests for `action_rules` package."""

import numpy as np
import pytest

from action_rules.candidates.candidate_generator import CandidateGenerator
from action_rules.rules.rules import Rules


@pytest.fixture
def candidate_generator():
    """
    Fixture to initialize a CandidateGenerator object with preset parameters.

    Returns
    -------
    CandidateGenerator
        An instance of the CandidateGenerator class.
    """
    frames = {0: np.array([[1, 0], [0, 1]]), 1: np.array([[0, 1], [1, 0]])}
    rules = Rules(undesired_state='0', desired_state='1', columns=['col1', 'col2'], count_transactions=2)
    return CandidateGenerator(
        frames=frames,
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_desired_support=1,
        min_undesired_confidence=0.5,
        min_desired_confidence=0.5,
        undesired_state=0,
        desired_state=1,
        rules=rules,
        use_sparse_matrix=False,
    )


def test_init(candidate_generator):
    """
    Test the initialization of the CandidateGenerator class.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that the initialization parameters are correctly set.
    """
    assert candidate_generator.min_stable_attributes == 1
    assert candidate_generator.min_flexible_attributes == 1
    assert candidate_generator.min_undesired_support == 1
    assert candidate_generator.min_undesired_confidence == 0.5
    assert candidate_generator.min_desired_support == 1
    assert candidate_generator.min_desired_confidence == 0.5
    assert candidate_generator.undesired_state == 0
    assert candidate_generator.desired_state == 1
    assert not candidate_generator.use_sparse_matrix


def test_generate_candidates(candidate_generator):
    """
    Test the generate_candidates method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that candidates are generated correctly.
    """
    ar_prefix = ()
    itemset_prefix = ()
    stable_items_binding = {'attr1': [0], 'attr2': [1]}
    flexible_items_binding = {'attr3': [0, 1]}
    undesired_mask = np.array([1, 0])
    desired_mask = np.array([0, 1])
    actionable_attributes = 1
    stop_list = []
    stop_list_itemset = []
    new_branches = candidate_generator.generate_candidates(
        ar_prefix,
        itemset_prefix,
        stable_items_binding,
        flexible_items_binding,
        undesired_mask,
        desired_mask,
        actionable_attributes,
        stop_list,
        stop_list_itemset,
        undesired_state=0,
        desired_state=1,
        verbose=False,
    )
    assert isinstance(new_branches, list)


def test_get_frames(candidate_generator):
    """
    Test the get_frames method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that frames for the undesired and desired states are returned correctly.
    """
    undesired_mask = np.array([1, 0])
    desired_mask = np.array([0, 1])
    undesired_frame, desired_frame = candidate_generator.get_frames(
        undesired_mask, desired_mask, undesired_state=0, desired_state=1
    )
    np.testing.assert_array_equal(undesired_frame, candidate_generator.frames[0] * undesired_mask)
    np.testing.assert_array_equal(desired_frame, candidate_generator.frames[1] * desired_mask)


def test_reduce_candidates_by_min_attributes(candidate_generator):
    """
    Test the reduce_candidates_by_min_attributes method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that the candidate sets are reduced correctly based on minimum attributes.
    """
    k = 1
    actionable_attributes = 1
    stable_items_binding = {'attr1': [0], 'attr2': [1]}
    flexible_items_binding = {'attr3': [0, 1]}
    reduced_stable, reduced_flexible = candidate_generator.reduce_candidates_by_min_attributes(
        k, actionable_attributes, stable_items_binding, flexible_items_binding
    )
    assert reduced_stable == {'attr1': [0], 'attr2': [1]}
    assert reduced_flexible == {}


def test_process_stable_candidates(candidate_generator):
    """
    Test the process_stable_candidates method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that stable candidates are processed correctly.
    """
    ar_prefix = ()
    itemset_prefix = ()
    reduced_stable_items_binding = {'attr1': [0]}
    stop_list = []
    stable_candidates = {'attr1': [0]}
    undesired_frame = np.array([[1, 0], [0, 1]])
    desired_frame = np.array([[0, 1], [1, 0]])
    new_branches = []
    candidate_generator.process_stable_candidates(
        ar_prefix,
        itemset_prefix,
        reduced_stable_items_binding,
        stop_list,
        stable_candidates,
        undesired_frame,
        desired_frame,
        new_branches,
        verbose=False,
    )
    assert isinstance(new_branches, list)


def test_process_flexible_candidates(candidate_generator):
    """
    Test the process_flexible_candidates method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that flexible candidates are processed correctly.
    """
    ar_prefix = ()
    itemset_prefix = ()
    reduced_flexible_items_binding = {'attr3': [0, 1]}
    stop_list = []
    stop_list_itemset = []
    flexible_candidates = {'attr3': [0, 1]}
    undesired_frame = np.array([[1, 0], [0, 1]])
    desired_frame = np.array([[0, 1], [1, 0]])
    actionable_attributes = 1
    new_branches = []
    candidate_generator.process_flexible_candidates(
        ar_prefix,
        itemset_prefix,
        reduced_flexible_items_binding,
        stop_list,
        stop_list_itemset,
        flexible_candidates,
        undesired_frame,
        desired_frame,
        actionable_attributes,
        new_branches,
        verbose=False,
    )
    assert isinstance(new_branches, list)


def test_process_items(candidate_generator):
    """
    Test the process_items method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that items are processed correctly to generate states and counts.
    """
    attribute = 'attr3'
    items = [0, 1]
    itemset_prefix = ()
    new_ar_prefix = ()
    stop_list_itemset = []
    undesired_frame = np.array([[1, 0], [0, 1]])
    desired_frame = np.array([[0, 1], [1, 0]])
    flexible_candidates = {'attr3': [0, 1]}
    verbose = False
    undesired_states, desired_states, undesired_count, desired_count = candidate_generator.process_items(
        attribute,
        items,
        itemset_prefix,
        new_ar_prefix,
        stop_list_itemset,
        undesired_frame,
        desired_frame,
        flexible_candidates,
        verbose,
    )
    assert isinstance(undesired_states, list)
    assert isinstance(desired_states, list)
    assert isinstance(undesired_count, int)
    assert isinstance(desired_count, int)


def test_update_new_branches(candidate_generator):
    """
    Test the update_new_branches method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that new branches are updated correctly.
    """
    new_branches = [{'item': 0}]
    stable_candidates = {'attr1': [0]}
    flexible_candidates = {'attr3': [0, 1]}
    candidate_generator.update_new_branches(new_branches, stable_candidates, flexible_candidates)
    assert 'stable_items_binding' in new_branches[0]
    assert 'flexible_items_binding' in new_branches[0]


def test_in_stop_list(candidate_generator):
    """
    Test the in_stop_list method.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
        The CandidateGenerator instance to test.

    Asserts
    -------
    Asserts that the stop list check is performed correctly.
    """
    ar_prefix = (0,)
    stop_list = [(0,)]
    result = candidate_generator.in_stop_list(ar_prefix, stop_list)
    assert result is True
    ar_prefix = (1,)
    result = candidate_generator.in_stop_list(ar_prefix, stop_list)
    assert result is False
