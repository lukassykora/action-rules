"""Class CandidateGenerator."""
import itertools
from typing import TYPE_CHECKING, Iterable, Optional, Union

from action_rules.rules import Rules

if TYPE_CHECKING:
    import cupy
    import numpy

# ``stop_list`` / ``stop_list_itemset`` accept either a list or a set;
# ``_add_stop_entry`` handles both. This alias keeps that polymorphism explicit.
StopList = Union[list, set]


class CandidateGenerator:
    """
    Generate candidate branches for bitset-based action-rule mining.

    Attributes
    ----------
    frames_bit_masks : dict
        Packed target-state masks keyed by target item index.
    bit_masks : Union[numpy.ndarray, cupy.ndarray]
        Packed masks for all one-hot items.
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

    Methods
    -------
    generate_candidates(ar_prefix, itemset_prefix, stable_items_binding, flexible_items_binding,
                        actionable_attributes, stop_list, stop_list_itemset)
        Generate candidate action rules.
    reduce_candidates_by_min_attributes(k, actionable_attributes, stable_items_binding, flexible_items_binding)
        Reduce the candidate sets based on minimum attributes.
    process_stable_candidates(ar_prefix, itemset_prefix, reduced_stable_items_binding, stop_list, stable_candidates,
                              new_branches)
        Process stable candidates to generate new branches.
    process_flexible_candidates(ar_prefix, itemset_prefix, reduced_flexible_items_binding, stop_list, stop_list_itemset,
                                flexible_candidates, actionable_attributes, new_branches)
        Process flexible candidates to generate new branches.
    process_items(attribute, items, itemset_prefix, stop_list_itemset, flexible_candidates)
        Process items to generate states and counts.
    update_new_branches(new_branches, stable_candidates, flexible_candidates)
        Update new branches with stable and flexible candidates.
    in_stop_list(ar_prefix, stop_list)
        Check if the action rule prefix is in the stop list.
    """

    _gpu_support_kernel_multi = None
    # GPU batches are sized to fit free device memory. A candidate "context" is two
    # packed masks of `num_words` uint64 words; each worklist item costs ~32 bytes of
    # kernel I/O (2x uint64 supports + int32/int64 indices, padded).
    _gpu_kernel_min_work = 512        # min (items x words) worth a kernel launch
    _gpu_bytes_per_word = 8           # uint64
    _gpu_support_bytes_per_item = 32
    _gpu_free_mem_fraction = 0.45     # share of currently-free VRAM one batch may use

    def __init__(
        self,
        min_stable_attributes: int,
        min_flexible_attributes: int,
        min_undesired_support: int,
        min_desired_support: int,
        min_undesired_confidence: float,
        min_desired_confidence: float,
        undesired_state: int,
        desired_state: int,
        rules: Rules,
        frames_bit_masks: Optional[dict] = None,
        bit_masks: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        verbose: bool = False,
    ):
        """
        Initialize the CandidateGenerator class with the specified parameters.

        Parameters
        ----------
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
        frames_bit_masks : dict, optional
            Packed bit-mask view of frames keyed by target item index.
        bit_masks : Union[numpy.ndarray, cupy.ndarray], optional
            Packed bit masks for all attributes (as produced by build_bit_masks).
        verbose : bool, optional
            If True, print per-candidate support diagnostics. Default is False.

        Notes
        -----
        The CandidateGenerator class is designed to facilitate the generation of candidate action rules by
        iterating over combinations of stable and flexible attributes. The class maintains a reference to the
        rules object where generated rules are stored.
        """
        self.frames_bit_masks = frames_bit_masks or {}
        self.bit_masks = bit_masks
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.undesired_state = undesired_state
        self.desired_state = desired_state
        self.rules = rules
        self.verbose = verbose

    def generate_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        stable_items_binding: dict,
        flexible_items_binding: dict,
        actionable_attributes: int,
        stop_list: StopList,
        stop_list_itemset: StopList,
        parent_undesired_mask: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        parent_desired_mask: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
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
        actionable_attributes : int
            Number of actionable attributes.
        stop_list : list
            List of stop combinations.
        stop_list_itemset : list
            List of stop itemsets.

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

        bitset_undesired_mask, bitset_desired_mask = self._resolve_bitset_masks(
            itemset_prefix,
            self.undesired_state,
            self.desired_state,
            parent_undesired_mask=parent_undesired_mask,
            parent_desired_mask=parent_desired_mask,
        )
        if bitset_undesired_mask is None or bitset_desired_mask is None:
            return []

        stable_candidates = {attribute: list(items) for attribute, items in stable_items_binding.items()}
        flexible_candidates = {attribute: list(items) for attribute, items in flexible_items_binding.items()}
        new_branches = []  # type: list

        self.process_stable_candidates(
            ar_prefix,
            itemset_prefix,
            reduced_stable_items_binding,
            stop_list,
            stable_candidates,
            new_branches,
            bitset_undesired_mask,
            bitset_desired_mask,
        )
        self.process_flexible_candidates(
            ar_prefix,
            itemset_prefix,
            reduced_flexible_items_binding,
            stop_list,
            stop_list_itemset,
            flexible_candidates,
            actionable_attributes,
            new_branches,
            undesired_mask_bitset=bitset_undesired_mask,
            desired_mask_bitset=bitset_desired_mask,
        )
        self.update_new_branches(new_branches, stable_candidates, flexible_candidates)

        return new_branches

    def generate_candidates_batch(
        self,
        candidates: list,
        stop_list: StopList,
        stop_list_itemset: StopList,
        batch_size: int = 32,
    ) -> list:
        """Generate candidates for a batch of branches, using GPU batching when possible."""
        if not candidates:
            return []
        if not self._can_use_gpu_batching():
            return self._generate_candidates_sequential(candidates, stop_list, stop_list_itemset)

        new_branches_all = []
        batch_contexts = []
        resolved_batch_size = max(1, int(batch_size))
        context_batch_size = self._gpu_context_batch_size(resolved_batch_size)

        for candidate in candidates:
            context = self._build_batch_context(candidate)
            if context is None:
                continue
            batch_contexts.append(context)
            if len(batch_contexts) >= context_batch_size:
                new_branches_all.extend(
                    self._flush_batch_contexts(batch_contexts, stop_list, stop_list_itemset)
                )
                batch_contexts = []

        if batch_contexts:
            new_branches_all.extend(
                self._flush_batch_contexts(batch_contexts, stop_list, stop_list_itemset)
            )

        return new_branches_all

    def _can_use_gpu_batching(self) -> bool:
        """Return True when candidate batching can run on GPU bit masks."""
        if self.verbose or self.bit_masks is None:
            return False
        return hasattr(self.bit_masks, "__cuda_array_interface__")

    def _build_batch_context(self, candidate: dict) -> Optional[dict]:
        """Prepare one context object used by the GPU batch expansion path."""
        bitset_undesired_mask, bitset_desired_mask = self._resolve_bitset_masks(
            candidate.get("itemset_prefix", tuple()),
            self.undesired_state,
            self.desired_state,
            parent_undesired_mask=candidate.get("parent_undesired_mask"),
            parent_desired_mask=candidate.get("parent_desired_mask"),
        )
        if bitset_undesired_mask is None or bitset_desired_mask is None:
            return None

        next_size = len(candidate["itemset_prefix"]) + 1
        reduced_stable_items_binding, reduced_flexible_items_binding = self.reduce_candidates_by_min_attributes(
            next_size,
            candidate["actionable_attributes"],
            candidate["stable_items_binding"],
            candidate["flexible_items_binding"],
        )
        return {
            "candidate": candidate,
            "ar_prefix": candidate["ar_prefix"],
            "itemset_prefix": candidate["itemset_prefix"],
            "reduced_stable_items_binding": reduced_stable_items_binding,
            "reduced_flexible_items_binding": reduced_flexible_items_binding,
            "stable_candidates": {
                attribute: list(items) for attribute, items in candidate["stable_items_binding"].items()
            },
            "flexible_candidates": {
                attribute: list(items) for attribute, items in candidate["flexible_items_binding"].items()
            },
            "bitset_undesired_mask": bitset_undesired_mask,
            "bitset_desired_mask": bitset_desired_mask,
            "actionable_attributes": candidate["actionable_attributes"],
        }

    def _flush_batch_contexts(
        self,
        batch_contexts: list,
        stop_list: StopList,
        stop_list_itemset: StopList,
    ) -> list:
        """Run one prepared context batch; fallback to sequential mode on failures."""
        batch_result = self._process_gpu_batch(batch_contexts, stop_list, stop_list_itemset)
        if batch_result is not None:
            return batch_result
        return self._generate_candidates_sequential(
            [context["candidate"] for context in batch_contexts],
            stop_list,
            stop_list_itemset,
        )

    def _generate_candidates_sequential(
        self,
        candidates: list,
        stop_list: StopList,
        stop_list_itemset: StopList,
    ) -> list:
        new_branches = []
        for candidate in candidates:
            new_branches.extend(
                self.generate_candidates(
                    **candidate,
                    stop_list=stop_list,
                    stop_list_itemset=stop_list_itemset,
                )
            )
        return new_branches

    def _process_gpu_batch(
        self,
        batch_contexts: list,
        stop_list: StopList,
        stop_list_itemset: StopList,
    ) -> Optional[list]:
        """
        Expand a batch of candidates in one GPU support pass.

        Returns the new branches, or None to signal the caller to retry the
        batch on the sequential CPU path.
        """
        try:
            import cupy as cp
        except ImportError:
            return None

        work_candidate_indices, work_item_indices = self._collect_gpu_worklist(
            batch_contexts, stop_list, stop_list_itemset
        )
        if not work_item_indices:
            return []

        branch_masks = self._stack_branch_masks(cp, batch_contexts)
        if branch_masks is None:
            return None

        supports = self._gpu_bitset_support_batch_multi(
            branch_masks[0],
            branch_masks[1],
            work_candidate_indices,
            work_item_indices,
        )
        if supports is None:
            return None
        undesired_supports_all, desired_supports_all = supports

        new_branches_all = []
        for context in batch_contexts:
            new_branches = self._expand_stable_slices(
                context, undesired_supports_all, desired_supports_all, stop_list
            )
            new_branches += self._expand_flex_slices(
                context, undesired_supports_all, desired_supports_all, stop_list, stop_list_itemset
            )
            self.update_new_branches(
                new_branches, context["stable_candidates"], context["flexible_candidates"]
            )
            new_branches_all.extend(new_branches)

        return new_branches_all

    def _collect_gpu_worklist(
        self,
        batch_contexts: list,
        stop_list: StopList,
        stop_list_itemset: StopList,
    ) -> tuple[list, list]:
        """
        Flatten active stable/flexible items across the batch into one worklist.

        Each context records its `stable_slices`/`flex_slices` (attribute, items,
        start offset) so kernel results can be scattered back per context.
        """
        work_candidate_indices: list = []
        work_item_indices: list = []
        for ctx_index, context in enumerate(batch_contexts):
            context["stable_slices"] = []
            context["flex_slices"] = []
            for attribute, items in context["reduced_stable_items_binding"].items():
                active_items = self._active_stable_items(context["ar_prefix"], items, stop_list)
                if not active_items:
                    continue
                start = len(work_item_indices)
                work_item_indices.extend(active_items)
                work_candidate_indices.extend([ctx_index] * len(active_items))
                context["stable_slices"].append((attribute, active_items, start))

            for attribute, items in context["reduced_flexible_items_binding"].items():
                new_ar_prefix = context["ar_prefix"] + (attribute,)
                if self.in_stop_list(new_ar_prefix, stop_list):
                    continue
                active_items = self._active_flexible_items(context["itemset_prefix"], items, stop_list_itemset)
                if not active_items:
                    continue
                start = len(work_item_indices)
                work_item_indices.extend(active_items)
                work_candidate_indices.extend([ctx_index] * len(active_items))
                context["flex_slices"].append((attribute, active_items, start))
        return work_candidate_indices, work_item_indices

    @staticmethod
    def _stack_branch_masks(cp, batch_contexts: list) -> Optional[tuple]:
        """Stack per-context packed masks into 2D arrays; None if stacking fails."""
        try:
            branch_masks_a = cp.stack(
                [cp.asarray(c["bitset_undesired_mask"], dtype=cp.uint64).reshape(-1) for c in batch_contexts],
                axis=0,
            )
            branch_masks_b = cp.stack(
                [cp.asarray(c["bitset_desired_mask"], dtype=cp.uint64).reshape(-1) for c in batch_contexts],
                axis=0,
            )
        except Exception:
            return None
        return branch_masks_a, branch_masks_b

    def _expand_stable_slices(
        self,
        context: dict,
        undesired_supports_all: list,
        desired_supports_all: list,
        stop_list: StopList,
    ) -> list:
        """Turn kernel supports for stable items into new branches / stop entries."""
        new_branches: list = []
        stable_candidates = context["stable_candidates"]
        for attribute, items, start in context["stable_slices"]:
            for offset, item in enumerate(items):
                new_ar_prefix = context["ar_prefix"] + (item,)
                if self.in_stop_list(new_ar_prefix, stop_list):
                    continue
                index = start + offset
                undesired_support = undesired_supports_all[index]
                desired_support = desired_supports_all[index]
                if undesired_support < self.min_undesired_support or desired_support < self.min_desired_support:
                    stable_candidates[attribute].remove(item)
                    self._add_stop_entry(stop_list, new_ar_prefix)
                else:
                    new_branches.append(
                        {
                            "ar_prefix": new_ar_prefix,
                            "itemset_prefix": new_ar_prefix,
                            "item": item,
                            "actionable_attributes": 0,
                            "parent_undesired_mask": context["bitset_undesired_mask"],
                            "parent_desired_mask": context["bitset_desired_mask"],
                        }
                    )
        return new_branches

    def _expand_flex_slices(
        self,
        context: dict,
        undesired_supports_all: list,
        desired_supports_all: list,
        stop_list: StopList,
        stop_list_itemset: StopList,
    ) -> list:
        """Turn kernel supports for flexible items into new branches and classification rules."""
        new_branches: list = []
        flexible_candidates = context["flexible_candidates"]
        for attribute, items, start in context["flex_slices"]:
            new_ar_prefix = context["ar_prefix"] + (attribute,)
            if self.in_stop_list(new_ar_prefix, stop_list):
                continue
            undesired_states = []
            desired_states = []
            undesired_count = 0
            desired_count = 0
            kept_items = []
            for offset, item in enumerate(items):
                if self.in_stop_list(context["itemset_prefix"] + (item,), stop_list_itemset):
                    continue
                index = start + offset
                undesired_support = undesired_supports_all[index]
                desired_support = desired_supports_all[index]

                undesired_conf = self.rules.calculate_confidence(undesired_support, desired_support)
                if undesired_support >= self.min_undesired_support:
                    undesired_count += 1
                    if undesired_conf >= self.min_undesired_confidence:
                        undesired_states.append(
                            {
                                "item": item,
                                "support": undesired_support,
                                "confidence": undesired_conf,
                            }
                        )
                    else:
                        self.rules.add_prefix_without_conf(new_ar_prefix, False)

                desired_conf = self.rules.calculate_confidence(desired_support, undesired_support)
                if desired_support >= self.min_desired_support:
                    desired_count += 1
                    if desired_conf >= self.min_desired_confidence:
                        desired_states.append(
                            {
                                "item": item,
                                "support": desired_support,
                                "confidence": desired_conf,
                            }
                        )
                    else:
                        self.rules.add_prefix_without_conf(new_ar_prefix, True)

                if desired_support < self.min_desired_support and undesired_support < self.min_undesired_support:
                    flexible_candidates[attribute].remove(item)
                    self._add_stop_entry(stop_list_itemset, context["itemset_prefix"] + (item,))
                    continue

                kept_items.append(item)

            if context["actionable_attributes"] == 0 and (undesired_count == 0 or desired_count == 0):
                del flexible_candidates[attribute]
                self._add_stop_entry(stop_list, context["ar_prefix"] + (attribute,))
            else:
                for item in kept_items:
                    new_branches.append(
                        {
                            "ar_prefix": new_ar_prefix,
                            "itemset_prefix": context["itemset_prefix"] + (item,),
                            "item": item,
                            "actionable_attributes": context["actionable_attributes"] + 1,
                            "parent_undesired_mask": context["bitset_undesired_mask"],
                            "parent_desired_mask": context["bitset_desired_mask"],
                        }
                    )
                if context["actionable_attributes"] + 1 >= self.min_flexible_attributes:
                    self.rules.add_classification_rules(
                        new_ar_prefix,
                        context["itemset_prefix"],
                        undesired_states,
                        desired_states,
                    )
        return new_branches

    @staticmethod
    def _add_stop_entry(stop_collection, value: tuple) -> None:
        """Add a stop entry to a list or set without branching at call sites."""
        if hasattr(stop_collection, "add"):
            stop_collection.add(value)
        else:
            stop_collection.append(value)

    def _active_stable_items(self, ar_prefix: tuple, items: list, stop_list: StopList) -> list:
        """Return stable items not blocked by `stop_list`."""
        active_items = []
        for item in items:
            if self.in_stop_list(ar_prefix + (item,), stop_list):
                continue
            active_items.append(item)
        return active_items

    def _active_flexible_items(self, itemset_prefix: tuple, items: list, stop_list_itemset: StopList) -> list:
        """Return flexible items not blocked by `stop_list_itemset`."""
        active_items = []
        for item in items:
            if self.in_stop_list(itemset_prefix + (item,), stop_list_itemset):
                continue
            active_items.append(item)
        return active_items

    def _resolve_bitset_masks(
        self,
        itemset_prefix: tuple,
        undesired_state: int,
        desired_state: int,
        parent_undesired_mask: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        parent_desired_mask: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
    ) -> tuple:
        if self.bit_masks is None or not self.frames_bit_masks:
            return None, None
        base_undesired = self.frames_bit_masks.get(undesired_state)
        base_desired = self.frames_bit_masks.get(desired_state)
        if base_undesired is None or base_desired is None:
            return None, None

        # Fast path: extend parent masks by the last item only (one AND per side).
        # Parent masks are passed as references on child candidates, so the BFS pays
        # O(1) launches per prefix extension instead of O(len(prefix)).
        if (
            parent_undesired_mask is not None
            and parent_desired_mask is not None
            and itemset_prefix
        ):
            last_item_row = self.bit_masks[itemset_prefix[-1]]
            return (
                parent_undesired_mask & last_item_row,
                parent_desired_mask & last_item_row,
            )

        bitset_undesired_mask = base_undesired
        bitset_desired_mask = base_desired
        for item in itemset_prefix:
            item_row = self.bit_masks[item]
            bitset_undesired_mask = bitset_undesired_mask & item_row
            bitset_desired_mask = bitset_desired_mask & item_row
        return bitset_undesired_mask, bitset_desired_mask

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
        stable_key_count = number_of_stable_attributes
        if stable_key_count < 0:
            stable_key_count = len(stable_items_binding) + stable_key_count
            if stable_key_count < 0:
                stable_key_count = 0
        flexible_key_count = number_of_flexible_attributes
        if flexible_key_count < 0:
            flexible_key_count = len(flexible_items_binding) + flexible_key_count
            if flexible_key_count < 0:
                flexible_key_count = 0
        reduced_stable_items_binding = {
            key: stable_items_binding[key]
            for key in itertools.islice(stable_items_binding.keys(), stable_key_count)
        }
        reduced_flexible_items_binding = {
            key: flexible_items_binding[key]
            for key in itertools.islice(flexible_items_binding.keys(), flexible_key_count)
        }
        return reduced_stable_items_binding, reduced_flexible_items_binding

    def process_stable_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        reduced_stable_items_binding: dict,
        stop_list: StopList,
        stable_candidates: dict,
        new_branches: list,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
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
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current undesired branch.
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current desired branch.
        new_branches : list
            List of new branches generated.

        Notes
        -----
        This method processes stable candidates by iterating over the reduced stable items bindings.
        It generates new action rule prefixes and calculates support for the undesired and desired states.
        If the support values meet the minimum thresholds, new branches are created and added to the
        new branches list.
        """
        if undesired_mask_bitset is None or desired_mask_bitset is None:
            return
        for attribute, items in reduced_stable_items_binding.items():
            active_items = self._active_stable_items(ar_prefix, items, stop_list)
            if not active_items:
                continue
            undesired_supports = self._bitset_support_batch(undesired_mask_bitset, active_items)
            desired_supports = self._bitset_support_batch(desired_mask_bitset, active_items)
            item_iter = zip(active_items, undesired_supports, desired_supports)

            for item, undesired_support, desired_support in item_iter:
                new_ar_prefix = ar_prefix + (item,)

                if self.verbose:
                    print('SUPPORT for: ' + str(itemset_prefix + (item,)))
                    print('_________________________________________________')
                    print('- extended by stable attribute')
                    print('undesired state support: ' + str(undesired_support))
                    print('desired state support: ' + str(desired_support))
                    print('')

                if undesired_support < self.min_undesired_support or desired_support < self.min_desired_support:
                    stable_candidates[attribute].remove(item)
                    self._add_stop_entry(stop_list, new_ar_prefix)
                else:
                    new_branches.append(
                        {
                            'ar_prefix': new_ar_prefix,
                            'itemset_prefix': new_ar_prefix,
                            'item': item,
                            'actionable_attributes': 0,
                            'parent_undesired_mask': undesired_mask_bitset,
                            'parent_desired_mask': desired_mask_bitset,
                        }
                    )

    def get_support(
        self,
        mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'],
        item: int,
    ) -> int:
        """
        Calculate support for one item under the provided packed branch mask.

        Parameters
        ----------
        mask_bitset : Union[numpy.ndarray, cupy.ndarray]
            Packed branch mask representing currently surviving transactions.
        item : int
            Item row index in `self.bit_masks`.

        Returns
        -------
        int
            Support count for the given item.
        """
        return self._bitset_support(mask_bitset, item)

    def _bitset_support(
        self, mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'], item: int
    ) -> int:
        """Compute support using packed bit masks by intersecting with the given mask."""
        attribute_mask = self.bit_masks[item]  # type: ignore[index]
        intersection = attribute_mask & mask_bitset
        return self._popcount(intersection)

    @staticmethod
    def _contiguous_indexer(items) -> Optional[slice]:
        """Return a slice for contiguous item indices to avoid advanced indexing copies."""
        if items is None:
            return None
        try:
            item_count = len(items)
        except Exception:
            return None
        if item_count == 0:
            return None
        if isinstance(items, range):
            if items.step == 1:
                return slice(items.start, items.stop)
            return None
        try:
            first = items[0]
            last = items[-1]
        except Exception:
            return None
        if (last - first + 1) != len(items):
            return None
        for offset, value in enumerate(items):
            if value != first + offset:
                return None
        return slice(first, last + 1)

    def _bitset_support_batch(
        self, mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'], items: list
    ) -> list[int]:
        """Compute support for multiple items in one packed-mask pass."""
        if self.bit_masks is None or not items:
            return []
        indexer = self._contiguous_indexer(items)
        if indexer is None:
            intersections = self.bit_masks[items] & mask_bitset
        else:
            intersections = self.bit_masks[indexer] & mask_bitset
        return self._popcount_rows(intersections)

    @classmethod
    def _should_use_gpu_kernel(cls, mask_bitset, item_count: int) -> bool:
        """Use the CUDA kernel only when the batch is big enough to amortize launch overhead."""
        num_words = int(getattr(mask_bitset, "size", 0) or 0)
        if num_words <= 0:
            return True
        return (item_count * num_words) >= cls._gpu_kernel_min_work

    def _gpu_free_budget_bytes(self) -> int:
        """Bytes one GPU batch may use: a fraction of currently-free device memory (0 if unknown)."""
        try:
            import cupy as cp

            free_bytes, _ = cp.cuda.runtime.memGetInfo()
        except Exception:
            return 0
        return int(max(0, free_bytes) * self._gpu_free_mem_fraction)

    def _gpu_context_batch_size(self, requested: int) -> int:  # pragma: no cover
        """How many candidate contexts (each = two packed masks) fit the GPU budget."""
        requested = max(1, int(requested))
        if self.bit_masks is not None and self.bit_masks.ndim:
            num_words = int(self.bit_masks.shape[-1])
        else:
            num_words = 0
        budget = self._gpu_free_budget_bytes()
        if num_words <= 0 or budget <= 0:
            return requested
        per_context = 2 * num_words * self._gpu_bytes_per_word
        return max(1, min(requested, budget // per_context))

    @classmethod
    def _get_gpu_support_kernel_multi(cls):
        """Lazily compile and cache a CUDA kernel for multi-branch batched support."""
        if cls._gpu_support_kernel_multi is not None:
            return cls._gpu_support_kernel_multi

        try:
            import cupy as cp
        except ImportError:
            return None

        kernel_code = r"""
        extern "C" __global__
        void bitset_support_kernel_multi(
            const unsigned long long* item_masks,
            const unsigned long long* branch_masks_a,
            const unsigned long long* branch_masks_b,
            const int* candidate_indices,
            const long long* item_indices,
            int num_words,
            unsigned long long* out_support_a,
            unsigned long long* out_support_b
        ) {
            extern __shared__ unsigned int shared_counts[];
            const int work_index = blockIdx.x;
            const int thread_id = threadIdx.x;

            const int candidate_index = candidate_indices[work_index];
            const long long item_index = item_indices[work_index];
            const unsigned long long* row_ptr =
                item_masks + ((size_t)item_index * (size_t)num_words);
            const unsigned long long* branch_a =
                branch_masks_a + ((size_t)candidate_index * (size_t)num_words);
            const unsigned long long* branch_b =
                branch_masks_b + ((size_t)candidate_index * (size_t)num_words);

            unsigned int local_count_a = 0u;
            unsigned int local_count_b = 0u;
            for (int word_index = thread_id; word_index < num_words; word_index += blockDim.x) {
                unsigned long long word = row_ptr[word_index];
                local_count_a += (unsigned int)__popcll(word & branch_a[word_index]);
                local_count_b += (unsigned int)__popcll(word & branch_b[word_index]);
            }

            unsigned int* shared_a = shared_counts;
            unsigned int* shared_b = shared_counts + blockDim.x;
            shared_a[thread_id] = local_count_a;
            shared_b[thread_id] = local_count_b;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (thread_id < stride) {
                    shared_a[thread_id] += shared_a[thread_id + stride];
                    shared_b[thread_id] += shared_b[thread_id + stride];
                }
                __syncthreads();
            }

            if (thread_id == 0) {
                out_support_a[work_index] = (unsigned long long)shared_a[0];
                out_support_b[work_index] = (unsigned long long)shared_b[0];
            }
        }
        """

        try:
            cls._gpu_support_kernel_multi = cp.RawKernel(
                kernel_code, "bitset_support_kernel_multi"
            )
        except Exception:
            cls._gpu_support_kernel_multi = None
        return cls._gpu_support_kernel_multi

    def _gpu_bitset_support_batch_multi(
        self,
        branch_masks_a: Union['numpy.ndarray', 'cupy.ndarray'],
        branch_masks_b: Union['numpy.ndarray', 'cupy.ndarray'],
        work_candidate_indices: list,
        work_item_indices: list,
    ) -> Optional[tuple[list[int], list[int]]]:
        """Compute support for a worklist across multiple branch masks in one kernel."""
        if self.bit_masks is None or not work_item_indices:
            return None

        try:
            import cupy as cp
            import numpy as np
        except ImportError:
            return None

        if not hasattr(self.bit_masks, "__cuda_array_interface__"):
            return None

        kernel = self._get_gpu_support_kernel_multi()
        if kernel is None:
            return None

        try:
            branch_masks_a = cp.asarray(branch_masks_a, dtype=cp.uint64)
            branch_masks_b = cp.asarray(branch_masks_b, dtype=cp.uint64)
            branch_masks_a = cp.ascontiguousarray(branch_masks_a)
            branch_masks_b = cp.ascontiguousarray(branch_masks_b)
            if branch_masks_a.shape != branch_masks_b.shape:
                return None
            if branch_masks_a.ndim != 2:
                return None

            num_words = int(branch_masks_a.shape[1])
            if num_words <= 0:
                return None

            total_items = len(work_item_indices)
            if not self._should_use_gpu_kernel(branch_masks_a, total_items):
                return None

            threads_per_block = 256
            shared_mem_bytes = threads_per_block * cp.dtype(cp.uint32).itemsize * 2

            budget_bytes = self._gpu_free_budget_bytes()
            if budget_bytes <= 0:
                chunk_items = total_items
            else:
                context_bytes = 2 * num_words * int(branch_masks_a.shape[0]) * self._gpu_bytes_per_word
                available = budget_bytes - context_bytes
                chunk_items = (
                    max(1, min(total_items, available // self._gpu_support_bytes_per_item))
                    if available > 0
                    else 1
                )

            supports_host_a = np.empty(total_items, dtype=np.int64)
            supports_host_b = np.empty(total_items, dtype=np.int64)
            for start in range(0, total_items, chunk_items):
                stop = min(total_items, start + chunk_items)
                chunk_len = stop - start
                candidate_indices = cp.asarray(
                    work_candidate_indices[start:stop], dtype=cp.int32
                )
                item_indices = cp.asarray(work_item_indices[start:stop], dtype=cp.int64)
                supports_a = cp.zeros(chunk_len, dtype=cp.uint64)
                supports_b = cp.zeros(chunk_len, dtype=cp.uint64)
                kernel(
                    (chunk_len,),
                    (threads_per_block,),
                    (
                        self.bit_masks,
                        branch_masks_a,
                        branch_masks_b,
                        candidate_indices,
                        item_indices,
                        int(num_words),
                        supports_a,
                        supports_b,
                    ),
                    shared_mem=shared_mem_bytes,
                )
                supports_host_a[start:stop] = cp.asnumpy(supports_a).astype(np.int64, copy=False)
                supports_host_b[start:stop] = cp.asnumpy(supports_b).astype(np.int64, copy=False)

            return supports_host_a.tolist(), supports_host_b.tolist()
        except Exception:
            return None

    def _popcount(self, mask: Union['numpy.ndarray', 'cupy.ndarray']) -> int:
        """Count the number of set bits in the packed mask."""
        return self._popcount_rows(mask)[0]

    @staticmethod
    def _popcount_uint64_rows(array: "numpy.ndarray") -> list[int]:
        """Compute popcount per row for uint64 arrays without unpackbits."""
        import numpy as np

        x = array.astype(np.uint64, copy=True)
        x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        x += x >> np.uint64(8)
        x += x >> np.uint64(16)
        x += x >> np.uint64(32)
        counts = x & np.uint64(0x7F)
        return counts.sum(axis=1).astype(np.int64, copy=False).tolist()

    def _popcount_rows(self, masks: Union['numpy.ndarray', 'cupy.ndarray']) -> list[int]:
        """Count set bits row-wise for 1D/2D packed masks."""
        import numpy as np

        if hasattr(masks, "__cuda_array_interface__"):
            try:
                import cupy as cp
            except ImportError:
                pass
            else:
                gpu_masks = cp.asarray(masks, dtype=cp.uint64)
                if gpu_masks.ndim == 1:
                    gpu_masks = gpu_masks.reshape(1, -1)
                if hasattr(cp, "bitwise_count"):
                    counts = cp.bitwise_count(gpu_masks).sum(axis=1)  # type: ignore[attr-defined]
                elif hasattr(gpu_masks, "bit_count"):
                    counts = gpu_masks.bit_count().sum(axis=1)  # type: ignore[call-arg]
                else:
                    cpu_masks = cp.asnumpy(gpu_masks)
                    return self._popcount_uint64_rows(cpu_masks)
                if hasattr(counts, "__cuda_array_interface__"):
                    # Keep a single host sync for GPU counts.
                    return cp.asnumpy(counts).astype(np.int64, copy=False).tolist()
                return [int(value) for value in counts.tolist()]

        array = np.asarray(masks, dtype=np.uint64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if hasattr(np, "bitwise_count"):
            counts = np.bitwise_count(array).sum(axis=1)  # type: ignore[attr-defined]
        elif hasattr(array, "bit_count"):
            counts = array.bit_count().sum(axis=1)  # type: ignore[call-arg]
        else:
            return self._popcount_uint64_rows(array)
        return [int(value) for value in counts.tolist()]

    def process_flexible_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        reduced_flexible_items_binding: dict,
        stop_list: StopList,
        stop_list_itemset: StopList,
        flexible_candidates: dict,
        actionable_attributes: int,
        new_branches: list,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
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
        actionable_attributes : int
            Number of actionable attributes.
        new_branches : list
            List of new branches generated.
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current undesired branch.
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current desired branch.

        Notes
        -----
        This method processes flexible candidates by iterating over the reduced flexible items bindings.
        It generates new action rule prefixes and calculates support for the undesired and desired states.
        If the support values meet the minimum thresholds, new branches are created and added to the
        new branches list. The method also updates the rules with new classification rules if the
        number of actionable attributes meets the minimum required.
        """
        if undesired_mask_bitset is None or desired_mask_bitset is None:
            return
        for attribute, items in reduced_flexible_items_binding.items():
            new_ar_prefix = ar_prefix + (attribute,)
            if self.in_stop_list(new_ar_prefix, stop_list):
                continue

            (
                undesired_states,
                desired_states,
                undesired_count,
                desired_count,
                kept_items,
            ) = self.process_items(
                attribute,
                items,
                itemset_prefix,
                new_ar_prefix,
                stop_list_itemset,
                flexible_candidates,
                undesired_mask_bitset=undesired_mask_bitset,
                desired_mask_bitset=desired_mask_bitset,
            )

            if actionable_attributes == 0 and (undesired_count == 0 or desired_count == 0):
                del flexible_candidates[attribute]
                self._add_stop_entry(stop_list, ar_prefix + (attribute,))
            else:
                for item in kept_items:
                    new_branches.append(
                        {
                            'ar_prefix': new_ar_prefix,
                            'itemset_prefix': itemset_prefix + (item,),
                            'item': item,
                            'actionable_attributes': actionable_attributes + 1,
                            'parent_undesired_mask': undesired_mask_bitset,
                            'parent_desired_mask': desired_mask_bitset,
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
        stop_list_itemset: StopList,
        flexible_candidates: dict,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
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
        flexible_candidates : dict
            Dictionary containing flexible candidates.
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current undesired branch.
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current desired branch.

        Returns
        -------
        tuple
            Tuple containing undesired states, desired states, undesired count, desired count,
            and the list of items kept for branching.

        Notes
        -----
        This method processes items by iterating over the list of items for a given attribute. It calculates
        support and confidence values for both undesired and desired states, updating the rules with new
        classification rules if the confidence thresholds are met. The method also removes items that do not
        meet the minimum support thresholds from the flexible candidates and updates the stop list accordingly.
        """
        undesired_states: list = []
        desired_states: list = []
        undesired_count = 0
        desired_count = 0
        kept_items: list = []
        if undesired_mask_bitset is None or desired_mask_bitset is None:
            return undesired_states, desired_states, undesired_count, desired_count, kept_items
        active_items = self._active_flexible_items(itemset_prefix, items, stop_list_itemset)
        if not active_items:
            item_iter: Iterable = ()
        else:
            undesired_supports = self._bitset_support_batch(undesired_mask_bitset, active_items)
            desired_supports = self._bitset_support_batch(desired_mask_bitset, active_items)
            item_iter = zip(active_items, undesired_supports, desired_supports)

        for item, undesired_support, desired_support in item_iter:
            if self.verbose:
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
                self._add_stop_entry(stop_list_itemset, itemset_prefix + (item,))
                continue

            kept_items.append(item)

        return undesired_states, desired_states, undesired_count, desired_count, kept_items

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

    def in_stop_list(self, ar_prefix: tuple, stop_list: StopList) -> bool:
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
            self._add_stop_entry(stop_list, ar_prefix)
            return True
        return False
