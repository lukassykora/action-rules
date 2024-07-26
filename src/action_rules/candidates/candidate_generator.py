"""Class CandidateGenerator."""

import copy
from typing import TYPE_CHECKING, Union

from action_rules.rules import Rules

if TYPE_CHECKING:
    import cupy
    import cupyx
    import numpy
    import scipy


class CandidateGenerator:
    """
    A class used to generate candidate action rules for a given dataset.

    Attributes
    ----------
    frames : dict
        Dictionary containing data frames for undesired and desired states.
    min_stable_attributes : int
        Minimum number of stable attributes required.
    min_flexible_attributes : int
        Minimum number of flexible attributes required.
    min_undesired_support : int
        Minimum support for the undesired state.
    min_desired_support : int
        Minimum support for the desired state.
    min_undesired_confidence : float
        Minimum confidence for the undesired state.
    min_desired_confidence : float
        Minimum confidence for the desired state.
    undesired_state : int
        The undesired state of the target attribute.
    desired_state : int
        The desired state of the target attribute.
    rules : Rules
        Rules object to store the generated classification rules.
    use_sparse_matrix : bool
        If True, sparse matrices are used for calculations.

    Methods
    -------
    generate_candidates(ar_prefix, itemset_prefix, stable_items_binding, flexible_items_binding, undesired_mask,
                        desired_mask, actionable_attributes, stop_list, stop_list_itemset, undesired_state,
                        desired_state, verbose=False)
        Generate candidate action rules.
    get_frames(undesired_mask, desired_mask, undesired_state, desired_state)
        Get the frames for the undesired and desired states.
    reduce_candidates_by_min_attributes(k, actionable_attributes, stable_items_binding, flexible_items_binding)
        Reduce the candidate sets based on minimum attributes.
    process_stable_candidates(ar_prefix, itemset_prefix, reduced_stable_items_binding, stop_list, stable_candidates,
                              undesired_frame, desired_frame, new_branches, verbose)
        Process stable candidates to generate new branches.
    process_flexible_candidates(ar_prefix, itemset_prefix, reduced_flexible_items_binding, stop_list, stop_list_itemset,
                                flexible_candidates, undesired_frame, desired_frame, actionable_attributes,
                                new_branches, verbose)
        Process flexible candidates to generate new branches.
    process_items(attribute, items, itemset_prefix, stop_list_itemset, undesired_frame, desired_frame,
                  flexible_candidates, verbose)
        Process items to generate states and counts.
    update_new_branches(new_branches, stable_candidates, flexible_candidates)
        Update new branches with stable and flexible candidates.
    in_stop_list(ar_prefix, stop_list)
        Check if the action rule prefix is in the stop list.
    """

    def __init__(
        self,
        frames: dict,
        min_stable_attributes: int,
        min_flexible_attributes: int,
        min_undesired_support: int,
        min_desired_support: int,
        min_undesired_confidence: float,
        min_desired_confidence: float,
        undesired_state: int,
        desired_state: int,
        rules: Rules,
        use_sparse_matrix: bool,
    ):
        """
        Initialize the CandidateGenerator class with the specified parameters.

        Parameters
        ----------
        frames : dict
            Dictionary containing data frames for undesired and desired states.
        min_stable_attributes : int
            Minimum number of stable attributes required.
        min_flexible_attributes : int
            Minimum number of flexible attributes required.
        min_undesired_support : int
            Minimum support for the undesired state.
        min_desired_support : int
            Minimum support for the desired state.
        min_undesired_confidence : float
            Minimum confidence for the undesired state.
        min_desired_confidence : float
            Minimum confidence for the desired state.
        undesired_state : int
            The undesired state of the target attribute.
        desired_state : int
            The desired state of the target attribute.
        rules : Rules
            Rules object to store the generated classification rules.
        use_sparse_matrix : bool, optional
            If True, sparse matrices are used. Default is False.

        Notes
        -----
        The CandidateGenerator class is designed to facilitate the generation of candidate action rules by
        iterating over combinations of stable and flexible attributes. The class maintains a reference to the
        rules object where generated rules are stored and supports both dense and sparse matrix operations.
        """
        self.frames = frames
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.undesired_state = undesired_state
        self.desired_state = desired_state
        self.rules = rules
        self.use_sparse_matrix = use_sparse_matrix

    def generate_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        stable_items_binding: dict,
        flexible_items_binding: dict,
        undesired_mask: Union['numpy.ndarray', 'cupy.ndarray', None],
        desired_mask: Union['numpy.ndarray', 'cupy.ndarray', None],
        actionable_attributes: int,
        stop_list: list,
        stop_list_itemset: list,
        undesired_state: int,
        desired_state: int,
        verbose: bool = False,
    ) -> list:
        """
        Generate candidate action rules.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.
        undesired_mask : Union['numpy.ndarray', 'cupy.ndarray', None]
            Mask for the undesired state.
        desired_mask : Union['numpy.ndarray', 'cupy.ndarray', None]
            Mask for the desired state.
        actionable_attributes : int
            Number of actionable attributes.
        stop_list : list
            List of stop combinations.
        stop_list_itemset : list
            List of stop itemsets.
        undesired_state : int
            The undesired state of the target attribute.
        desired_state : int
            The desired state of the target attribute.
        verbose : bool, optional
            If True, enables verbose output. Default is False.

        Returns
        -------
        list
            List of new branches generated.

        Notes
        -----
        This method generates candidate action rules by processing both stable and flexible attributes.
        It first reduces the candidate sets based on the minimum attributes, then processes stable and
        flexible candidates to generate new branches. The new branches are updated with the candidates
        and returned.
        """
        k = len(itemset_prefix) + 1
        reduced_stable_items_binding, reduced_flexible_items_binding = self.reduce_candidates_by_min_attributes(
            k, actionable_attributes, stable_items_binding, flexible_items_binding
        )

        undesired_frame, desired_frame = self.get_frames(undesired_mask, desired_mask, undesired_state, desired_state)
        stable_candidates = copy.deepcopy(stable_items_binding)
        flexible_candidates = copy.deepcopy(flexible_items_binding)

        new_branches = []  # type: list

        self.process_stable_candidates(
            ar_prefix,
            itemset_prefix,
            reduced_stable_items_binding,
            stop_list,
            stable_candidates,
            undesired_frame,
            desired_frame,
            new_branches,
            verbose,
        )
        self.process_flexible_candidates(
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
            verbose,
        )
        self.update_new_branches(new_branches, stable_candidates, flexible_candidates)

        return new_branches

    def get_frames(
        self,
        undesired_mask: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix', None
        ],
        desired_mask: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix', None
        ],
        undesired_state: int,
        desired_state: int,
    ) -> tuple:
        """
        Get the frames for the undesired and desired states.

        Parameters
        ----------
        undesired_mask : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                               'scipy.sparse.csr_matrix', None]
            Mask for the undesired state.
        desired_mask : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                            'scipy.sparse.csr_matrix', None]
            Mask for the desired state.
        undesired_state : int
            The undesired state of the target attribute.
        desired_state : int
            The desired state of the target attribute.

        Returns
        -------
        tuple
            Tuple containing the frames for the undesired and desired states.

        Notes
        -----
        This method retrieves the frames for the undesired and desired states using the provided masks.
        If no masks are provided, the method returns the frames as they are stored. If masks are provided,
        the method applies them to the frames, performing element-wise multiplication to filter the data.
        """
        if undesired_mask is None:
            return self.frames[undesired_state], self.frames[desired_state]
        else:
            if self.use_sparse_matrix:
                if undesired_mask.getnnz() > 0:  # type: ignore
                    undesired_frame = self.frames[undesired_state].multiply(undesired_mask)
                else:
                    undesired_frame = self.frames[undesired_state] * 0
                if desired_mask.getnnz() > 0:  # type: ignore
                    desired_frame = self.frames[desired_state].multiply(desired_mask)
                else:
                    desired_frame = self.frames[desired_state] * 0
            else:
                undesired_frame = self.frames[undesired_state] * undesired_mask
                desired_frame = self.frames[desired_state] * desired_mask
            del undesired_mask, desired_mask
            return undesired_frame, desired_frame

    def reduce_candidates_by_min_attributes(
        self, k: int, actionable_attributes: int, stable_items_binding: dict, flexible_items_binding: dict
    ) -> tuple:
        """
        Reduce the candidate sets based on minimum attributes.

        Parameters
        ----------
        k : int
            Length of the itemset prefix plus one.
        actionable_attributes : int
            Number of actionable attributes.
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.

        Returns
        -------
        tuple
            Tuple containing the reduced stable and flexible items bindings.

        Notes
        -----
        This method reduces the candidate sets by removing attributes that do not meet the minimum
        number of stable or flexible attributes required. The reduction is based on the length of the
        itemset prefix plus one (k) and the number of actionable attributes.
        """
        number_of_stable_attributes = len(stable_items_binding) - (self.min_stable_attributes - k)
        if k > self.min_stable_attributes:
            number_of_flexible_attributes = len(flexible_items_binding) - (
                self.min_flexible_attributes - actionable_attributes - 1
            )
        else:
            number_of_flexible_attributes = 0
        reduced_stable_items_binding = {
            k: stable_items_binding[k] for k in list(stable_items_binding.keys())[:number_of_stable_attributes]
        }
        reduced_flexible_items_binding = {
            k: flexible_items_binding[k] for k in list(flexible_items_binding.keys())[:number_of_flexible_attributes]
        }
        return reduced_stable_items_binding, reduced_flexible_items_binding

    def process_stable_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        reduced_stable_items_binding: dict,
        stop_list: list,
        stable_candidates: dict,
        undesired_frame: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'
        ],
        desired_frame: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'
        ],
        new_branches: list,
        verbose: bool,
    ):
        """
        Process stable candidates to generate new branches.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        reduced_stable_items_binding : dict
            Dictionary containing reduced bindings for stable items.
        stop_list : list
            List of stop combinations.
        stable_candidates : dict
            Dictionary containing stable candidates.
        undesired_frame : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                                'scipy.sparse.csr_matrix']
            Data frame for the undesired state.
        desired_frame : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                              'scipy.sparse.csr_matrix']
            Data frame for the desired state.
        new_branches : list
            List of new branches generated.
        verbose : bool
            If True, enables verbose output.

        Notes
        -----
        This method processes stable candidates by iterating over the reduced stable items bindings.
        It generates new action rule prefixes and calculates support for the undesired and desired states.
        If the support values meet the minimum thresholds, new branches are created and added to the
        new branches list.
        """
        for attribute, items in reduced_stable_items_binding.items():
            for item in items:
                new_ar_prefix = ar_prefix + (item,)
                if self.in_stop_list(new_ar_prefix, stop_list):
                    continue

                undesired_support = self.get_support(undesired_frame, item)
                desired_support = self.get_support(desired_frame, item)

                if verbose:
                    print('SUPPORT for: ' + str(itemset_prefix + (item,)))
                    print('_________________________________________________')
                    print('- extended by stable attribute')
                    print('undesired state support: ' + str(undesired_support))
                    print('desired state support: ' + str(desired_support))
                    print('')

                if undesired_support < self.min_undesired_support or desired_support < self.min_desired_support:
                    stable_candidates[attribute].remove(item)
                    stop_list.append(new_ar_prefix)
                else:
                    new_branches.append(
                        {
                            'ar_prefix': new_ar_prefix,
                            'itemset_prefix': new_ar_prefix,
                            'item': item,
                            'undesired_mask': undesired_frame[item],
                            'desired_mask': desired_frame[item],
                            'actionable_attributes': 0,
                        }
                    )

    def get_support(
        self,
        frame: Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'],
        item: int,
    ) -> int:
        """
        Calculate the sum of elements in a specified row of the given frame.

        This function takes a 2D array (`frame`) and an integer (`item`) representing
        the index of a row. It returns the sum of all elements in that row.

        Parameters
        ----------
        frame : Union[numpy.ndarray, cupy.ndarray, 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix']
            A 2D array from which the row is selected. The array can be a NumPy or CuPy array.
        item : int
            The index of the row to be summed.

        Returns
        -------
        int
            The sum of all elements in the specified row of the frame.

        Notes
        -----
        - This function is compatible with both NumPy and CuPy arrays.
        - Ensure that the `item` index is within the bounds of the frame's rows.
        - For sparse matrices, the sum is computed efficiently by leveraging sparse matrix operations.
        """
        return frame[item].sum()

    def process_flexible_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        reduced_flexible_items_binding: dict,
        stop_list: list,
        stop_list_itemset: list,
        flexible_candidates: dict,
        undesired_frame: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'
        ],
        desired_frame: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'
        ],
        actionable_attributes: int,
        new_branches: list,
        verbose: bool,
    ):
        """
        Process flexible candidates to generate new branches.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        reduced_flexible_items_binding : dict
            Dictionary containing reduced bindings for flexible items.
        stop_list : list
            List of stop combinations.
        stop_list_itemset : list
            List of stop itemsets.
        flexible_candidates : dict
            Dictionary containing flexible candidates.
        undesired_frame : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                                'scipy.sparse.csr_matrix']
            Data frame for the undesired state.
        desired_frame : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                              'scipy.sparse.csr_matrix']
            Data frame for the desired state.
        actionable_attributes : int
            Number of actionable attributes.
        new_branches : list
            List of new branches generated.
        verbose : bool
            If True, enables verbose output.

        Notes
        -----
        This method processes flexible candidates by iterating over the reduced flexible items bindings.
        It generates new action rule prefixes and calculates support for the undesired and desired states.
        If the support values meet the minimum thresholds, new branches are created and added to the
        new branches list. The method also updates the rules with new classification rules if the
        number of actionable attributes meets the minimum required.
        """
        for attribute, items in reduced_flexible_items_binding.items():
            new_ar_prefix = ar_prefix + (attribute,)
            if self.in_stop_list(new_ar_prefix, stop_list):
                continue

            undesired_states, desired_states, undesired_count, desired_count = self.process_items(
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

            if actionable_attributes == 0 and (undesired_count == 0 or desired_count == 0):
                del flexible_candidates[attribute]
                stop_list.append(ar_prefix + (attribute,))
            else:
                for item in items:
                    new_branches.append(
                        {
                            'ar_prefix': new_ar_prefix,
                            'itemset_prefix': itemset_prefix + (item,),
                            'item': item,
                            'undesired_mask': undesired_frame[item],
                            'desired_mask': desired_frame[item],
                            'actionable_attributes': actionable_attributes + 1,
                        }
                    )
                if actionable_attributes + 1 >= self.min_flexible_attributes:
                    self.rules.add_classification_rules(
                        new_ar_prefix,
                        itemset_prefix,
                        undesired_states,
                        desired_states,
                    )

    def process_items(
        self,
        attribute: str,
        items: list,
        itemset_prefix: tuple,
        new_ar_prefix: tuple,
        stop_list_itemset: list,
        undesired_frame: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'
        ],
        desired_frame: Union[
            'numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix'
        ],
        flexible_candidates: dict,
        verbose: bool,
    ):
        """
        Process items to generate states and counts.

        Parameters
        ----------
        attribute : str
            The attribute being processed.
        items : list
            List of items for the attribute.
        itemset_prefix : tuple
            Prefix of the itemset.
        new_ar_prefix : tuple
            Prefix for stop list.
        stop_list_itemset : list
            List of stop itemsets.
        undesired_frame : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                                'scipy.sparse.csr_matrix']
            Data frame for the undesired state.
        desired_frame : Union['numpy.ndarray', 'cupy.ndarray', 'cupyx.scipy.sparse.csr_matrix',
                              'scipy.sparse.csr_matrix']
            Data frame for the desired state.
        flexible_candidates : dict
            Dictionary containing flexible candidates.
        verbose : bool
            If True, enables verbose output.

        Returns
        -------
        tuple
            Tuple containing undesired states, desired states, undesired count, and desired count.

        Notes
        -----
        This method processes items by iterating over the list of items for a given attribute. It calculates
        support and confidence values for both undesired and desired states, updating the rules with new
        classification rules if the confidence thresholds are met. The method also removes items that do not
        meet the minimum support thresholds from the flexible candidates and updates the stop list accordingly.
        """
        undesired_states = []
        desired_states = []
        undesired_count = 0
        desired_count = 0

        for item in items:
            if self.in_stop_list(itemset_prefix + (item,), stop_list_itemset):
                continue

            undesired_support = undesired_frame[item].sum()
            desired_support = desired_frame[item].sum()

            if verbose:
                print('SUPPORT for: ' + str(itemset_prefix + (item,)))
                print('_________________________________________________')
                print('- extended by flexible attribute')
                print('undesired state support: ' + str(undesired_support))
                print('desired state support: ' + str(desired_support))
                print('')

            undesired_conf = self.rules.calculate_confidence(undesired_support, desired_support)
            if undesired_support >= self.min_undesired_support:
                undesired_count += 1
                if undesired_conf >= self.min_undesired_confidence:
                    undesired_states.append({'item': item, 'support': undesired_support, 'confidence': undesired_conf})
                else:
                    self.rules.add_prefix_without_conf(new_ar_prefix, False)

            desired_conf = self.rules.calculate_confidence(desired_support, undesired_support)
            if desired_support >= self.min_desired_support:
                desired_count += 1
                if desired_conf >= self.min_desired_confidence:
                    desired_states.append({'item': item, 'support': desired_support, 'confidence': desired_conf})
                else:
                    self.rules.add_prefix_without_conf(new_ar_prefix, True)

            if desired_support < self.min_desired_support and undesired_support < self.min_undesired_support:
                flexible_candidates[attribute].remove(item)
                stop_list_itemset.append(itemset_prefix + (item,))

        return undesired_states, desired_states, undesired_count, desired_count

    def update_new_branches(self, new_branches: list, stable_candidates: dict, flexible_candidates: dict):
        """
        Update new branches with stable and flexible candidates.

        Parameters
        ----------
        new_branches : list
            List of new branches generated.
        stable_candidates : dict
            Dictionary containing stable candidates.
        flexible_candidates : dict
            Dictionary containing flexible candidates.

        Notes
        -----
        This method updates new branches by iterating over stable and flexible candidates. It creates
        new stable and flexible bindings for each new branch, ensuring that only the relevant candidates
        are included in the new branches.
        """
        for new_branch in new_branches:
            adding = False
            new_stable = {}  # type: dict
            new_flexible = {}  # type: dict

            for attribute, items in stable_candidates.items():
                for item in items:
                    if adding:
                        if attribute not in new_stable:
                            new_stable[attribute] = []
                        new_stable[attribute].append(item)
                    if item == new_branch['item']:
                        adding = True

            for attribute, items in flexible_candidates.items():
                for item in items:
                    if adding:
                        if attribute not in new_flexible:
                            new_flexible[attribute] = []
                        new_flexible[attribute].append(item)
                    if item == new_branch['item']:
                        adding = True

            del new_branch['item']
            new_branch['stable_items_binding'] = new_stable
            new_branch['flexible_items_binding'] = new_flexible

    def in_stop_list(self, ar_prefix: tuple, stop_list: list) -> bool:
        """
        Check if the action rule prefix is in the stop list.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        stop_list : list
            List of stop combinations.

        Returns
        -------
        bool
            True if the action rule prefix is in the stop list, False otherwise.

        Notes
        -----
        This method checks if the action rule prefix is in the stop list by checking for the presence
        of the last two elements and all but the first element of the prefix in the stop list. If the
        prefix is found, it is added to the stop list to prevent future processing.
        """
        if ar_prefix[-2:] in stop_list:
            return True
        if ar_prefix[1:] in stop_list:
            stop_list.append(ar_prefix)
            return True
        return False
