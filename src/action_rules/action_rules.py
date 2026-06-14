"""Main class ActionRules."""

import itertools
import warnings
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Optional, Union  # noqa

from .candidates.candidate_generator import CandidateGenerator
from .output.output import Output
from .rules.rules import Rules

if TYPE_CHECKING:
    from types import ModuleType  # noqa

    import cudf
    import cupy
    import numpy
    import pandas


class ActionRules:
    """
    Generate action rules from tabular data using one-hot encoding and bitset support counting.

    Attributes
    ----------
    min_stable_attributes : int
        The minimum number of stable attributes required.
    min_flexible_attributes : int
        The minimum number of flexible attributes required.
    min_undesired_support : int
        The minimum support for the undesired state.
    min_undesired_confidence : float
        The minimum confidence for the undesired state.
    min_desired_support : int
        The minimum support for the desired state.
    min_desired_confidence : float
        The minimum confidence for the desired state.
    verbose : bool, optional
        If True, enables verbose output.
    rules : Optional[Rules], optional
        Stores the generated rules.
    output : Optional[Output], optional
        Stores the generated action rules.
    np : Optional[ModuleType], optional
        The numpy or cupy module used for array operations.
    pd : Optional[ModuleType], optional
        The pandas or cudf module used for DataFrame operations.
    is_gpu_np : bool
        Indicates whether GPU-accelerated numpy (cupy) is used.
    is_gpu_pd : bool
        Indicates whether GPU-accelerated pandas (cudf) is used.
    intrinsic_utility_table : dict, optional
        (attribute, value) -> float
        A lookup table for the intrinsic utility of each attribute-value pair.
        If None, no intrinsic utility is considered.
    transition_utility_table : dict, optional
        (attribute, from_value, to_value) -> float
        A lookup table for cost/gain of transitions between values.
        If None, no transition utility is considered.

    Methods
    -------
    fit(data, stable_attributes, flexible_attributes, target, undesired_state, desired_state, use_gpu=False)
        Generates action rules based on the provided dataset and parameters.
    get_bindings(data, stable_attributes, flexible_attributes, target)
        Binds attributes to corresponding columns in the dataset.
    get_stop_list(stable_items_binding, flexible_items_binding)
        Generates a stop list to prevent certain combinations of attributes.
    get_rules()
        Returns the generated action rules if available.
    predict(frame_row)
        Predicts recommended actions based on the provided row of data.
    """

    def __init__(
        self,
        min_stable_attributes: int,
        min_flexible_attributes: int,
        min_undesired_support: int,
        min_undesired_confidence: float,
        min_desired_support: int,
        min_desired_confidence: float,
        verbose=False,
        intrinsic_utility_table: Optional[dict] = None,
        transition_utility_table: Optional[dict] = None,
    ):
        """
        Initialize the ActionRules class with the specified parameters.

        Parameters
        ----------
        min_stable_attributes : int
            The minimum number of stable attributes required.
        min_flexible_attributes : int
            The minimum number of flexible attributes required.
        min_undesired_support : int
            The minimum support for the undesired state.
        min_undesired_confidence : float
            The minimum confidence for the undesired state.
        min_desired_support : int
            The minimum support for the desired state.
        min_desired_confidence : float
            The minimum confidence for the desired state.
        verbose : bool, optional
            If True, enables verbose output. Default is False.
        intrinsic_utility_table : dict, optional
            (attribute, value) -> float
            A lookup table for the intrinsic utility of each attribute-value pair.
            If None, no intrinsic utility is considered.
        transition_utility_table : dict, optional
            (attribute, from_value, to_value) -> float
            A lookup table for cost/gain of transitions between values.
            If None, no transition utility is considered.

        Notes
        -----
        The `verbose` parameter can be used to enable detailed output during the rule generation process.
        """
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.verbose = verbose
        self.rules = None  # type: Optional[Rules]
        self.output = None  # type: Optional[Output]
        self.np = None  # type: Optional[ModuleType]
        self.pd = None  # type: Optional[ModuleType]
        self.is_gpu_np = False
        self.is_gpu_pd = False
        self.is_onehot = False
        self.bit_masks = None  # type: Optional['numpy.ndarray']
        self.target_state_bit_masks = None  # type: Optional[dict]
        self.frames_bit_masks = None  # type: Optional[dict]
        self.intrinsic_utility_table = intrinsic_utility_table or {}
        self.transition_utility_table = transition_utility_table or {}
        self._original_intrinsic_utility_table = {}  # type: dict
        self._original_transition_utility_table = {}  # type: dict
        self._column_values = None  # type: Optional[dict]

    def count_max_nodes(self, stable_items_binding: dict, flexible_items_binding: dict) -> int:
        """
        Calculate the maximum number of nodes based on the given item bindings.

        This function takes two dictionaries, `stable_items_binding` and `flexible_items_binding`,
        which map attributes to lists of items. It calculates the total number of nodes by considering
        all possible combinations of the lengths of these item lists and summing the product of each combination.

        Parameters
        ----------
        stable_items_binding : dict
            A dictionary where keys are attributes and values are lists of stable items.
        flexible_items_binding : dict
            A dictionary where keys are attributes and values are lists of flexible items.

        Returns
        -------
        int
            The total number of nodes calculated by summing the product of lengths of all combinations of item lists.

        Notes
        -----
        - The function first combines the lengths of item lists from both dictionaries.
        - It then calculates the sum of the products of all possible combinations of these lengths.
        """
        import numpy

        values_in_attribute = []
        for items in list(stable_items_binding.values()) + list(flexible_items_binding.values()):
            values_in_attribute.append(len(items))

        sum_nodes = 0
        for i in range(len(values_in_attribute)):
            for comb in itertools.combinations(values_in_attribute, i + 1):
                sum_nodes += int(numpy.prod(comb))
        return sum_nodes

    def set_array_library(self, use_gpu: bool, df: Union['cudf.DataFrame', 'pandas.DataFrame']):
        """
        Set the appropriate array and DataFrame libraries (cuDF or pandas) based on the user's preference.

        Parameters
        ----------
        use_gpu : bool
            Indicates whether to use GPU (cuDF) for data processing if available.
        df : Union[cudf.DataFrame, pandas.DataFrame]
            The DataFrame to convert.

        Raises
        ------
        ImportError
            If `use_gpu` is True but cuDF is not available and pandas cannot be imported as fallback.

        Warnings
        --------
        UserWarning
            If `use_gpu` is True but cuDF is not available, a warning is issued indicating fallback to pandas.

        Notes
        -----
        This method determines whether to use GPU-accelerated libraries for processing data, falling back to CPU-based
        libraries if necessary.
        """
        if use_gpu:
            try:
                import cupy as np

                is_gpu_np = True
            except ImportError:
                warnings.warn("CuPy is not available. Falling back to Numpy.")
                import numpy as np

                is_gpu_np = False
        else:
            import numpy as np

            is_gpu_np = False

        df_library_imported = False
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                is_gpu_pd = False
                df_library_imported = True
        except ImportError:
            df_library_imported = False

        if not df_library_imported:
            try:
                import cudf as pd

                if isinstance(df, pd.DataFrame):
                    is_gpu_pd = True
                    df_library_imported = True
            except ImportError:
                df_library_imported = False

        if not df_library_imported:
            raise ImportError('Just Pandas or cuDF dataframes are supported.')

        self.np = np
        self.pd = pd
        self.is_gpu_np = is_gpu_np
        self.is_gpu_pd = is_gpu_pd

    def df_to_array(self, df: Union['cudf.DataFrame', 'pandas.DataFrame']) -> tuple:
        """
        Convert a one-hot DataFrame to a binary array.

        Parameters
        ----------
        df : Union[cudf.DataFrame, pandas.DataFrame]
            The DataFrame to convert.

        Returns
        -------
        tuple
            A tuple containing the transposed array and the DataFrame columns.

        Notes
        -----
        The data is converted to an unsigned 8-bit array (`np.uint8`), backed by
        NumPy or CuPy depending on the selected cpu/gpu backend.
        """
        columns = list(df.columns)
        if self.is_gpu_np:
            data = self.np.asarray(df.values, dtype=self.np.uint8).T  # type: ignore
        elif self.is_gpu_pd:
            data = df.to_numpy().T  # type: ignore
        else:
            data = df.to_numpy(dtype=self.np.uint8).T  # type: ignore
        return data, columns

    def build_bit_masks(
        self,
        data: Union['numpy.ndarray', 'cupy.ndarray'],
    ) -> Union['numpy.ndarray', 'cupy.ndarray']:
        """
        Pack a binary feature matrix into 64-bit masks for fast intersection.

        Parameters
        ----------
        data : Union[numpy.ndarray, cupy.ndarray]
            Dense matrix produced by `df_to_array`, shaped (num_attributes, num_transactions)
            and containing 0/1 values.

        Returns
        -------
        Union[numpy.ndarray, cupy.ndarray]
            bit_masks is a uint64 array with shape (num_attributes, num_words)
            holding 64 bits/transactions for each item.

        Notes
        -----
        - The packing uses 64-bit little-endian words (bit 0 corresponds to the
          first transaction in each chunk).
        - Sparse inputs are not supported; callers should densify before packing.
        """
        if self.np is None:
            raise RuntimeError("Array library is not initialised. Call set_array_library first.")
        # Shape is (num_attributes, num_transactions).
        num_attributes, num_transactions = data.shape
        num_words = (num_transactions + 63) // 64
        padded_transactions = num_words * 64
        padding = padded_transactions - num_transactions

        if padding > 0:
            pad_block = self.np.zeros((num_attributes, padding), dtype=data.dtype)
            padded_data = self.np.concatenate((data, pad_block), axis=1)
        else:
            padded_data = data

        # Group transactions into 64-bit chunks: (num_attributes, num_words, 64).
        chunks = padded_data.reshape(num_attributes, num_words, 64).astype(self.np.uint64, copy=False)
        bit_offsets = self.np.arange(64, dtype=self.np.uint64)
        bit_weights = self.np.left_shift(self.np.uint64(1), bit_offsets)

        # Pack each 64-sized transaction chunk into one uint64 word.
        bit_masks = self.np.tensordot(chunks, bit_weights, axes=([2], [0])).astype(self.np.uint64, copy=False)
        return bit_masks

    def _cache_bitset_structures(
        self,
        bit_masks: Union['numpy.ndarray', 'cupy.ndarray'],
        target_items_binding: dict,
        target: str,
    ) -> None:
        """
        Save all column masks; extract target-state rows into a separate dict.

        Parameters
        ----------
        bit_masks : Union[numpy.ndarray, cupy.ndarray]
            Packed transaction masks for every attribute/value.
        target_items_binding : dict
            Mapping from target attribute name to indices of its one-hot columns.
        target : str
            Name of the target attribute.
        """
        target_state_indices = target_items_binding.get(target, [])
        target_state_bit_masks = {index: bit_masks[index] for index in target_state_indices}

        self.bit_masks = bit_masks
        self.target_state_bit_masks = target_state_bit_masks

    def one_hot_encode(
        self,
        data: Union['cudf.DataFrame', 'pandas.DataFrame'],
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
    ) -> Union['cudf.DataFrame', 'pandas.DataFrame']:
        """
        Perform one-hot encoding on the attributes of the DataFrame.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The input DataFrame containing the data to be encoded.
        stable_attributes : list
            List of stable attributes to be one-hot encoded.
        flexible_attributes : list
            List of flexible attributes to be one-hot encoded.
        target : str
            The target attribute to be one-hot encoded.

        Returns
        -------
        Union[cudf.DataFrame, pandas.DataFrame]
            A DataFrame with the specified attributes one-hot encoded.

        Notes
        -----
        Stable and flexible (antecedent) columns are cast to strings only for non-missing values; ``NaN`` is
        preserved so that ``pd.get_dummies`` skips it instead of creating a phantom ``<attr>_<item_*>_nan``
        category.  This implements the *pessimistic* interpretation of null values in incomplete information
        systems --- a missing antecedent does not match any value-specific itemset and therefore cannot appear
        in a discovered rule --- as defined for action-rule mining by Dardzinska, *Action Rules Mining*
        (Springer 2013, Section 2.3.2).  The target column is cast to strings in full so that any ``NaN``
        target value becomes its own explicit category (downstream ``get_split_bit_masks`` will exclude it
        from both the undesired and desired splits, which is the intended behaviour when callers want to
        ignore unlabelled rows).
        """

        def _prepare_antecedent_frame(frame, attributes):
            """Stringify non-missing antecedent cells while keeping ``NaN`` as ``NaN``.

            Letting ``get_dummies`` see a real ``NaN`` is the documented way to make it skip the value;
            calling ``astype(str)`` first would convert ``np.nan`` into the literal string ``'nan'`` and
            spawn a spurious one-hot column.
            """
            antecedent = frame[attributes].copy()
            return antecedent.where(antecedent.isna(), antecedent.astype(str))

        to_concat = []
        if len(stable_attributes) > 0:
            stable_frame = _prepare_antecedent_frame(data, stable_attributes)
            data_stable = self.pd.get_dummies(stable_frame, sparse=False, prefix_sep='_<item_stable>_')  # type: ignore
            to_concat.append(data_stable)
        if len(flexible_attributes) > 0:
            flexible_frame = _prepare_antecedent_frame(data, flexible_attributes)
            data_flexible = self.pd.get_dummies(  # type: ignore
                flexible_frame, sparse=False, prefix_sep='_<item_flexible>_'
            )
            to_concat.append(data_flexible)
        data_target = self.pd.get_dummies(  # type: ignore
            data[[target]].astype(str), sparse=False, prefix_sep='_<item_target>_'
        )
        to_concat.append(data_target)
        data = self.pd.concat(to_concat, axis=1)  # type: ignore
        return data

    def fit_onehot(
        self,
        data: Union['cudf.DataFrame', 'pandas.DataFrame'],
        stable_attributes: dict,
        flexible_attributes: dict,
        target: dict,
        target_undesired_state: str,
        target_desired_state: str,
        use_sparse_matrix: bool = False,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        Fit the model when input data is already one-hot encoded.

        The method remaps one-hot columns to the internal naming convention
        (`_<item_stable>_`, `_<item_flexible>_`, `_<item_target>_`), drops
        unrelated columns, and forwards execution to `fit`.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The dataset to be processed and used for fitting the model.
        stable_attributes : dict
            A dictionary mapping stable attribute names to lists of column
            names corresponding to those attributes.
        flexible_attributes : dict
            A dictionary mapping flexible attribute names to lists of column
            names corresponding to those attributes.
        target : dict
            A dictionary mapping the target attribute name to a list of
            column names corresponding to that attribute.
        target_undesired_state : str
            The undesired state of the target attribute, used in action rule generation.
        target_desired_state : str
            The desired state of the target attribute, used in action rule generation.
        use_sparse_matrix : bool, optional
            Kept for backward compatibility with action-rules <= 1.0.11. The bitset
            backend supersedes sparse matrices, so this flag is accepted and ignored.
            Other unrecognized keyword arguments (``**kwargs``) are likewise accepted
            and ignored for backward compatibility with older call signatures.
        use_gpu : bool, optional
            If True, the GPU (cuDF) is used for data processing if available.
            Default is False.

        Notes
        -----
        This method expects boolean/binary one-hot columns.
        """
        self.is_onehot = True
        data = data.copy()
        data = data.astype('bool')
        new_labels = []
        attributes_stable = set([])
        attribtes_flexible = set([])
        attribute_target = ''
        remove_cols = []
        for label in data.columns:
            to_remove = True
            for attribute, columns in stable_attributes.items():
                if label in columns:
                    new_labels.append(attribute + '_<item_stable>_' + label)
                    attributes_stable.add(attribute)
                    to_remove = False
            for attribute, columns in flexible_attributes.items():
                if label in columns:
                    new_labels.append(attribute + '_<item_flexible>_' + label)
                    attribtes_flexible.add(attribute)
                    to_remove = False
            for attribute, columns in target.items():
                if label in columns:
                    new_labels.append(attribute + '_<item_target>_' + label)
                    attribute_target = attribute
                    to_remove = False
            if to_remove:
                new_labels.append(label)
                remove_cols.append(label)
        data.columns = new_labels
        data = data.drop(columns=remove_cols)
        self.fit(
            data,
            list(attributes_stable),
            list(attribtes_flexible),
            attribute_target,
            target_undesired_state,
            target_desired_state,
            use_sparse_matrix=use_sparse_matrix,
            use_gpu=use_gpu,
        )

    def fit(
        self,
        data: Union['cudf.DataFrame', 'pandas.DataFrame'],
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
        target_undesired_state: str,
        target_desired_state: str,
        use_sparse_matrix: bool = False,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        Generate action rules for the provided dataset.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The dataset to generate action rules from.
        stable_attributes : list
            List of stable attributes.
        flexible_attributes : list
            List of flexible attributes.
        target : str
            The target attribute.
        target_undesired_state : str
            The undesired state of the target attribute.
        target_desired_state : str
            The desired state of the target attribute.
        use_sparse_matrix : bool, optional
            Kept for backward compatibility with action-rules <= 1.0.11. The bitset
            backend supersedes sparse matrices, so this flag is accepted and ignored.
            Other unrecognized keyword arguments (``**kwargs``) are likewise accepted
            and ignored for backward compatibility with older call signatures.
        use_gpu : bool, optional
            Use GPU (cuDF) for data processing if available. Default is False.

        Raises
        ------
        RuntimeError
            If the model has already been fitted.

        Notes
        -----
        The method runs one-hot encoding (when needed), packs bit masks, explores
        candidate branches, prunes classification rules by depth, and finally
        materializes action rules.
        """
        if self.output is not None:
            raise RuntimeError("The model is already fit.")
        if use_sparse_matrix:
            warnings.warn(
                "The 'use_sparse_matrix' parameter is obsolete and has no effect: action-rules now "
                "always uses the packed-bitset backend.",
                UserWarning,
                stacklevel=2,
            )
        # Forward tolerance: legacy callers passed use_gpu="auto" for backend
        # autoselection. That harness lives outside the package now, so treat any
        # truthy string as a plain GPU request instead of raising.
        if isinstance(use_gpu, str):
            use_gpu = use_gpu.strip().lower() not in ("", "false", "cpu", "no", "0")

        # reset cached bitset structures before fitting a new model
        self.bit_masks = None
        self.target_state_bit_masks = None
        self.frames_bit_masks = None
        self.set_array_library(use_gpu, data)
        if not self.is_onehot:
            data = self.one_hot_encode(data, stable_attributes, flexible_attributes, target)
        data, columns = self.df_to_array(data)

        stable_items_binding, flexible_items_binding, target_items_binding, column_values = self.get_bindings(
            columns, stable_attributes, flexible_attributes, target
        )

        # Preserve original string-keyed tables before remapping to integer indices.
        # confidence_intervals() needs the originals to pass to inference engines.
        self._original_intrinsic_utility_table = dict(self.intrinsic_utility_table)
        self._original_transition_utility_table = dict(self.transition_utility_table)
        self._column_values = column_values
        self.intrinsic_utility_table, self.transition_utility_table = self.remap_utility_tables(column_values)

        local_bit_masks = self.build_bit_masks(data)
        self._cache_bitset_structures(local_bit_masks, target_items_binding, target)
        self.frames_bit_masks = self.get_split_bit_masks(target_items_binding, target)

        if self.verbose:
            print('Maximum number of nodes to check for support:')
            print('_____________________________________________')
            print(self.count_max_nodes(stable_items_binding, flexible_items_binding))
            print('')
        use_gpu_batching = bool(self.is_gpu_np and self.bit_masks is not None and self.frames_bit_masks)

        # Set membership is hot in candidate pruning; use a set internally for O(1) lookups.
        stop_list = set(self.get_stop_list(stable_items_binding, flexible_items_binding))
        undesired_state = columns.index(target + '_<item_target>_' + str(target_undesired_state))
        desired_state = columns.index(target + '_<item_target>_' + str(target_desired_state))

        stop_list_itemset = set()  # type: set

        initial_candidate = {
            'ar_prefix': tuple(),
            'itemset_prefix': tuple(),
            'stable_items_binding': stable_items_binding,
            'flexible_items_binding': flexible_items_binding,
            'actionable_attributes': 0,
        }
        candidates_pool = deque([initial_candidate])
        pending_depth_counts = {0: 1}
        min_pending_depth: Optional[int] = 0
        max_depth_seen = 0
        next_prune_depth = 1
        self.rules = Rules(
            undesired_state,
            desired_state,
            columns,
            data.shape[1],
            self.intrinsic_utility_table,
            self.transition_utility_table,
        )
        candidate_generator = CandidateGenerator(
            frames_bit_masks=self.frames_bit_masks,
            bit_masks=self.bit_masks,
            min_stable_attributes=self.min_stable_attributes,
            min_flexible_attributes=self.min_flexible_attributes,
            min_undesired_support=self.min_undesired_support,
            min_desired_support=self.min_desired_support,
            min_undesired_confidence=self.min_undesired_confidence,
            min_desired_confidence=self.min_desired_confidence,
            undesired_state=undesired_state,
            desired_state=desired_state,
            rules=self.rules,
            verbose=self.verbose,
        )
        # Default GPU node batch; the adaptive VRAM budgeting in CandidateGenerator
        # shrinks this automatically to fit available device memory.
        effective_gpu_node_batch_size = 32

        def pop_next_candidate() -> dict:
            """Pop one pending candidate and keep pending-depth bookkeeping in sync."""
            nonlocal min_pending_depth
            candidate_to_expand = candidates_pool.popleft()
            depth = len(candidate_to_expand['ar_prefix'])
            pending_depth_counts[depth] -= 1
            if pending_depth_counts[depth] <= 0:
                pending_depth_counts.pop(depth, None)
                if depth == min_pending_depth:
                    min_pending_depth = min(pending_depth_counts.keys(), default=None)
            return candidate_to_expand

        while len(candidates_pool) > 0:
            if use_gpu_batching:  # pragma: no cover
                batch: list = []
                while candidates_pool and len(batch) < effective_gpu_node_batch_size:
                    batch.append(pop_next_candidate())
                new_candidates = candidate_generator.generate_candidates_batch(
                    batch,
                    stop_list=stop_list,
                    stop_list_itemset=stop_list_itemset,
                    batch_size=effective_gpu_node_batch_size,
                )
            else:
                candidate = pop_next_candidate()
                new_candidates = candidate_generator.generate_candidates(
                    **candidate,
                    stop_list=stop_list,
                    stop_list_itemset=stop_list_itemset,
                )
            if new_candidates:
                candidates_pool.extend(new_candidates)
                for new_candidate in new_candidates:
                    new_depth = len(new_candidate['ar_prefix'])
                    pending_depth_counts[new_depth] = pending_depth_counts.get(new_depth, 0) + 1
                    if min_pending_depth is None or new_depth < min_pending_depth:
                        min_pending_depth = new_depth
                    if new_depth > max_depth_seen:
                        max_depth_seen = new_depth
            while next_prune_depth <= max_depth_seen and (
                min_pending_depth is None or min_pending_depth >= next_prune_depth
            ):
                self.rules.prune_classification_rules(next_prune_depth, stop_list)
                next_prune_depth += 1
        self.rules.generate_action_rules()
        self.output = Output(
            self.rules.action_rules, target, stable_items_binding, flexible_items_binding, column_values
        )
        del data
        if self.is_gpu_np:  # pragma: no cover
            gpu_pool = self.np.get_default_memory_pool()  # type: ignore[union-attr, attr-defined]
            gpu_pool.free_all_blocks()

    def get_bindings(
        self,
        columns: list,
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
    ) -> tuple:
        """
        Bind stable/flexible/target attribute to corresponding column in the dataset.

        Parameters
        ----------
        columns : list
            List of column names in the dataset.
        stable_attributes : list
            List of stable attributes.
        flexible_attributes : list
            List of flexible attributes.
        target : str
            The target attribute.

        Returns
        -------
        tuple
            A tuple containing the bindings for stable attributes, flexible attributes, and target items.

        Notes
        -----
        The method generates mappings from column indices to attribute values for stable, flexible, and target
        attributes.
        """
        stable_items_binding = defaultdict(lambda: [])
        flexible_items_binding = defaultdict(lambda: [])
        target_items_binding = defaultdict(lambda: [])
        column_values = {}

        for i, col in enumerate(columns):
            is_continue = False
            # stable
            for attribute in stable_attributes:
                if col.startswith(attribute + '_<item_stable>_'):
                    stable_items_binding[attribute].append(i)
                    column_values[i] = (attribute, col.split('_<item_stable>_', 1)[1])
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # flexible
            for attribute in flexible_attributes:
                if col.startswith(attribute + '_<item_flexible>_'):
                    flexible_items_binding[attribute].append(i)
                    column_values[i] = (attribute, col.split('_<item_flexible>_', 1)[1])
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # target
            if col.startswith(target + '_<item_target>_'):
                target_items_binding[target].append(i)
                column_values[i] = (target, col.split('_<item_target>_', 1)[1])
        return stable_items_binding, flexible_items_binding, target_items_binding, column_values

    def get_stop_list(self, stable_items_binding: dict, flexible_items_binding: dict) -> list:
        """
        Generate a stop list to prevent certain combinations of attributes.

        Parameters
        ----------
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.

        Returns
        -------
        list
            A list of stop combinations.

        Notes
        -----
        The stop list is generated by creating pairs of stable item indices and ensuring flexible items do not repeat.
        """
        stop_list = []
        for items in stable_items_binding.values():
            for stop_couple in itertools.product(items, repeat=2):
                stop_list.append(tuple(stop_couple))
        for item in flexible_items_binding.keys():
            stop_list.append(tuple([item, item]))
        return stop_list

    def get_split_bit_masks(self, target_items_binding: dict, target: str) -> dict:
        """
        Return packed bit-mask rows for each target state.

        Parameters
        ----------
        target_items_binding : dict
            Indexes of target attributes columns in one-hot table.
        target : str
            Name of the target attribute.

        Returns
        -------
        dict
            Dictionary mapping target attributes to the corresponding packed mask rows.

        Notes
        -----
        Requires that `build_bit_masks` has been executed beforehand.
        """
        if self.bit_masks is None:
            raise RuntimeError("Bit masks are not available. Ensure fit() was run first.")

        target_state_masks = {}
        for item_index in target_items_binding.get(target, []):
            target_state_masks[item_index] = self.bit_masks[item_index]
        return target_state_masks

    def get_rules(self) -> Output:
        """
        Return the generated action rules if available.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Returns
        -------
        Output
            The generated action rules.

        Notes
        -----
        This method returns the `Output` object containing the generated action rules.
        """
        if self.output is None:
            raise RuntimeError("The model is not fit.")
        return self.output

    def predict(self, frame_row: Union['cudf.Series', 'pandas.Series']) -> Union['cudf.DataFrame', 'pandas.DataFrame']:
        """
        Predict recommended actions based on the provided row of data.

        This method applies the fitted action rules to the given row of data and generates
        a DataFrame with recommended actions if any of the action rules are triggered.

        Parameters
        ----------
        frame_row : Union['cudf.Series', 'pandas.Series']
            A row of data in the form of a cuDF or pandas Series. The Series should
            contain the features required by the action rules.

        Returns
        -------
        Union['cudf.DataFrame', 'pandas.DataFrame']
            A DataFrame with the recommended actions. The DataFrame includes the following columns:
            - The original attributes with recommended changes.
            - 'ActionRules_RuleIndex': Index of the action rule applied.
            - 'ActionRules_UndesiredSupport': Support of the undesired part of the rule.
            - 'ActionRules_DesiredSupport': Support of the desired part of the rule.
            - 'ActionRules_UndesiredConfidence': Confidence of the undesired part of the rule.
            - 'ActionRules_DesiredConfidence': Confidence of the desired part of the rule.
            - 'ActionRules_Uplift': Uplift value of the rule.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Notes
        -----
        The method compares the given row of data against the undesired itemsets of the action rules.
        If a match is found, it applies the desired itemset changes and records the action rule's
        metadata. The result is a DataFrame with one or more rows representing the recommended actions
        for the given data.
        """
        if self.output is None:
            raise RuntimeError("The model is not fit.")
        index_value_tuples = list(zip(frame_row.index, frame_row))
        values = []
        column_values = self.output.column_values
        for index_value_tuple in index_value_tuples:
            values.append(list(column_values.keys())[list(column_values.values()).index(index_value_tuple)])
        new_values = tuple(values)
        predicted = []
        for i, action_rule in enumerate(self.output.action_rules):
            if set(action_rule['undesired']['itemset']) <= set(new_values):
                predicted_row = frame_row.copy()
                for recommended in set(action_rule['desired']['itemset']) - set(new_values):
                    attribute, value = column_values[recommended]
                    predicted_row[attribute + ' (Recommended)'] = value
                predicted_row['ActionRules_RuleIndex'] = i
                predicted_row['ActionRules_UndesiredSupport'] = action_rule['undesired']['support']
                predicted_row['ActionRules_DesiredSupport'] = action_rule['desired']['support']
                predicted_row['ActionRules_UndesiredConfidence'] = action_rule['undesired']['confidence']
                predicted_row['ActionRules_DesiredConfidence'] = action_rule['desired']['confidence']
                predicted_row['ActionRules_Uplift'] = action_rule['uplift']
                predicted.append(predicted_row)
        return self.pd.DataFrame(predicted)  # type: ignore

    def confidence_intervals(
        self,
        data,
        method: str = "bootstrap",
        confidence_level: float = 0.95,
        threshold: Optional[float] = None,
        metric: str = "uplift",
        n_bootstrap: int = 1000,
        n_mc: int = 10000,
        random_state: Optional[int] = None,
        analytic_type: str = "wald",
        bootstrap_type: str = "percentile",
    ):
        r"""Compute confidence intervals for all fitted action rules.

        Applies a statistical inference engine to the provided dataset and
        attaches confidence interval results to the output object.  Results
        are also returned directly for immediate inspection.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The original (pre-encoding) dataset used for inference.  Columns
            must match the attribute names supplied during ``fit()``.
        method : str, optional
            CI method to use.  One of:

            - ``'bootstrap'`` — non-parametric percentile bootstrap
              (default).
            - ``'analytic'`` or ``'wald'`` — closed-form Wald interval via
              the delta method (requires ``scipy``).
            - ``'bayesian'`` — Beta-Binomial conjugate model with Monte
              Carlo posterior sampling.
        confidence_level : float, optional
            Nominal coverage probability, e.g. ``0.95`` (default).
        threshold : float, optional
            Decision boundary used to categorise rules after computing
            intervals.  When ``None`` (default), categorisation is skipped.
        metric : str, optional
            Metric to use for categorisation when *threshold* is provided.
            One of ``'uplift'`` (default) or ``'realistic_rule_gain'``.
        n_bootstrap : int, optional
            Number of bootstrap resamples.  Only used when
            ``method='bootstrap'``.  Default ``1000``.
        n_mc : int, optional
            Number of Monte Carlo samples.  Only used when
            ``method='bayesian'``.  Default ``10000``.
        random_state : int, optional
            Seed for reproducibility.  Passed to the engine when applicable.
            ``None`` uses the global NumPy random state.
        analytic_type : str, optional
            Sub-type of the analytic method.  Only used when
            ``method='analytic'`` or ``method='wald'``.  One of:

            - ``'wald'`` — standard Wald normal approximation (default).
            - ``'newcombe_wilson'`` (preferred) or ``'wilson'`` (alias) —
              Newcombe-Wilson interval (Newcombe, 1998) for the unscaled
              rule contrast :math:`\\delta = p_d + p_u - 1`, built by
              combining two single-proportion Wilson score intervals; the
              resulting interval is asymmetric.  ``'wilson'`` is retained
              as a backward-compatible alias.
            - ``'auto'`` — Newcombe-Wilson when sample is small (``n < 40``)
              or proportion is extreme (``< 0.05`` or ``> 0.95``), Wald
              otherwise (following Agresti & Coull, 1998).
        bootstrap_type : str, optional
            Sub-type of the bootstrap method.  Only used when
            ``method='bootstrap'``.  One of:

            - ``'percentile'`` — standard percentile bootstrap (default).
            - ``'bca'`` — bias-corrected and accelerated (BCa) interval,
              which adjusts for bias and skewness using jackknife
              acceleration (Efron, 1987).

        Returns
        -------
        list
            List of :class:`~action_rules.inference.base.ConfidenceIntervalResult`
            objects, one per action rule, in the same order as
            ``self.output.action_rules``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet (``self.output is None``).
        ValueError
            If *method* is not one of the supported values.

        Notes
        -----
        Results are also stored on the output object via
        ``self.output.set_confidence_intervals(results)`` so that subsequent
        calls to ``get_ar_notation()``, ``get_pretty_ar_notation()``, and
        ``get_export_notation()`` include the CI information.
        """
        if self.output is None:
            raise RuntimeError("The model is not fit.")

        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1.")
        if n_mc < 1:
            raise ValueError("n_mc must be >= 1.")
        valid_analytic_types = {"wald", "wilson", "newcombe_wilson", "auto"}
        if analytic_type not in valid_analytic_types:
            raise ValueError(f"Unknown analytic_type '{analytic_type}'. Choose from {valid_analytic_types}.")
        valid_bootstrap_types = {"percentile", "bca"}
        if bootstrap_type not in valid_bootstrap_types:
            raise ValueError(f"Unknown bootstrap_type '{bootstrap_type}'. Choose from {valid_bootstrap_types}.")
        valid_metrics = {"uplift", "realistic_rule_gain"}
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {valid_metrics}.")

        from .inference.base import categorize_rule, extract_rule_masks

        masks = extract_rule_masks(self.output)

        engine: Any
        if method == "bootstrap":
            from .inference.bootstrap import BootstrapEngine

            engine = BootstrapEngine(n_bootstrap, random_state, bootstrap_type=bootstrap_type)
        elif method in ("analytic", "wald"):
            from .inference.analytic import AnalyticEngine

            engine = AnalyticEngine(analytic_type=analytic_type)
        elif method == "bayesian":
            from .inference.bayesian import BayesianEngine

            engine = BayesianEngine(n_mc, random_state=random_state)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Supported methods: 'bootstrap', 'analytic', 'wald', 'bayesian'."
            )

        results = engine.compute(
            data=data,
            rules=masks,
            confidence_level=confidence_level,
            intrinsic_utility_table=self._original_intrinsic_utility_table or None,
            transition_utility_table=self._original_transition_utility_table or None,
            column_values=self._column_values,
        )

        if threshold is not None:
            for result in results:
                if metric == "uplift":
                    result.category = categorize_rule(result.uplift_ci_lower, result.uplift_ci_upper, threshold)
                else:
                    if (
                        result.realistic_rule_gain_ci_lower is not None
                        and result.realistic_rule_gain_ci_upper is not None
                    ):
                        result.category = categorize_rule(
                            result.realistic_rule_gain_ci_lower,
                            result.realistic_rule_gain_ci_upper,
                            threshold,
                        )

        self.output.set_confidence_intervals(results)
        return results

    def cross_validate(
        self,
        data,
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
        target_undesired_state: str,
        target_desired_state: str,
        *,
        n_splits: int = 5,
        stratify: bool = True,
        strategies=None,
        metrics=None,
        k_fraction: float = 0.2,
        ci_method: str = 'bootstrap',
        n_bootstrap: int = 500,
        risk_lambda: float = 1.96,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        n_bootstrap_oof: int = 1000,
        bootstrap_design: str = 'cluster_fold',
        track_stability: bool = True,
        use_sparse_matrix: bool = False,
        scale_support_thresholds: bool = True,
        compute_insample_baseline: bool = False,
    ):
        """Run stratified K-fold cross-validation on the action-rule pipeline.

        Each fold receives a fresh :class:`ActionRules` instance configured
        with the same hyperparameters and utility tables as ``self``.  Per
        fold, rules are mined on the train split, confidence intervals are
        computed on the train split, and every discovered rule is re-scored
        on the held-out test split (``test_uplift``, ``test_realistic_gain``).
        Targeting metrics (``uplift@k``, Qini, AUUC, ``profit@k``) are
        evaluated under several targeting strategies on the test split.

        Parameters
        ----------
        data : pandas.DataFrame
            The full dataset, pre-encoding.
        stable_attributes, flexible_attributes : list of str
        target : str
        target_undesired_state, target_desired_state : str
        n_splits : int, optional
            Number of folds (default ``5``).  Must be ≥ 2.
        stratify : bool, optional
            Whether to stratify folds by ``target`` value (default ``True``).
        strategies : sequence of str, optional
            Subset of ``('point', 'lower', 'lower_positive', 'risk_adjusted')``.
            Defaults to all four.
        metrics : sequence of str, optional
            Subset of ``('uplift_at_k', 'qini', 'auuc', 'profit_at_k')``.
            Defaults to all four.
        k_fraction : float, optional
            Top-k cutoff used by the ``*_at_k`` metrics (default ``0.2``).
        ci_method, n_bootstrap, confidence_level, risk_lambda : forwarded to
            :class:`~action_rules.evaluation.cv.CrossValidator`.
        random_state : int, optional
            Seed for fold splitting and bootstrap CIs.
        n_bootstrap_oof : int, optional
            Bootstrap replicates for the across-fold rule-resampling CI.
            Set to ``0`` to disable bootstrap CIs.
        bootstrap_design : str, optional
            ``'cluster_fold'`` (default) resamples rules within each fold,
            computes the metric per fold, and averages — so the bootstrap CI
            estimates the same fold-mean quantity as the ``mean`` column.
            ``'oof_pool'`` (legacy) resamples within fold then concatenates
            into one pool before computing the metric; estimates a pool-level
            statistic that differs from the fold mean by roughly a factor of
            K.
        track_stability : bool, optional
            Compute pairwise Jaccard overlap of discovered rule sets across
            folds (default ``True``).
        compute_insample_baseline : bool, optional
            When ``True``, additionally mine on the full dataset and score on
            the full dataset to compute an apparent (in-sample) performance
            baseline; the result is stored on
            ``CrossValidationResult.insample_summary``.  Default ``False``
            preserves the existing return shape.

        Returns
        -------
        action_rules.evaluation.cv.CrossValidationResult

        Notes
        -----
        - Naive K-fold CIs based on ``mean ± 1.96·std/√K`` over folds have
          below-nominal coverage (Bates, Hastie & Tibshirani, 2021,
          arXiv:2104.00673).  This method therefore reports fold spread
          (``std``) as a stability indicator and a stratified bootstrap CI
          over OOF rule records as the inferential interval.
        - Calling :meth:`cross_validate` does **not** require the model to
          be fitted on the full data first.  It does not mutate ``self``;
          each fold operates on a fresh internal instance.
        """
        from .evaluation.cv import METRICS, STRATEGIES, CrossValidator

        # Snapshot the hyperparameters needed to build pristine per-fold instances.
        min_stable_attributes = self.min_stable_attributes
        min_flexible_attributes = self.min_flexible_attributes
        # Support thresholds are absolute counts.  When mining on a train fold
        # that is ``(n_splits-1)/n_splits`` of the full data, scale them down
        # proportionally so the same prevalence requirements apply on each fold.
        scale = (n_splits - 1) / n_splits if scale_support_thresholds else 1.0
        min_undesired_support = max(1, int(round(self.min_undesired_support * scale)))
        min_desired_support = max(1, int(round(self.min_desired_support * scale)))
        min_undesired_confidence = self.min_undesired_confidence
        min_desired_confidence = self.min_desired_confidence
        verbose = self.verbose
        intrinsic = self._original_intrinsic_utility_table or dict(self.intrinsic_utility_table)
        transition = self._original_transition_utility_table or dict(self.transition_utility_table)

        def _factory():
            return ActionRules(
                min_stable_attributes=min_stable_attributes,
                min_flexible_attributes=min_flexible_attributes,
                min_undesired_support=min_undesired_support,
                min_undesired_confidence=min_undesired_confidence,
                min_desired_support=min_desired_support,
                min_desired_confidence=min_desired_confidence,
                verbose=verbose,
                intrinsic_utility_table=intrinsic or None,
                transition_utility_table=transition or None,
            )

        validator = CrossValidator(
            _factory,
            stable_attributes=stable_attributes,
            flexible_attributes=flexible_attributes,
            target=target,
            target_undesired_state=target_undesired_state,
            target_desired_state=target_desired_state,
            n_splits=n_splits,
            stratify=stratify,
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            strategies=STRATEGIES if strategies is None else strategies,
            metrics=METRICS if metrics is None else metrics,
            k_fraction=k_fraction,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            risk_lambda=risk_lambda,
            confidence_level=confidence_level,
            random_state=random_state,
            n_bootstrap_oof=n_bootstrap_oof,
            bootstrap_design=bootstrap_design,
            track_stability=track_stability,
            use_sparse_matrix=use_sparse_matrix,
            compute_insample_baseline=compute_insample_baseline,
        )
        return validator.run(data)

    def remap_utility_tables(self, column_values):
        """
        Remap the keys of intrinsic and transition utility tables using the provided column mapping.

        The function uses `column_values`, a dictionary mapping internal column indices to
        (attribute, value) tuples, to invert the mapping so that utility table keys are replaced
        with the corresponding integer index (for intrinsic utilities) or a tuple of integer indices
        (for transition utilities).

        Parameters
        ----------
        column_values : dict
            Dictionary mapping integer column indices to (attribute, value) pairs.
            Example: {0: ('Age', 'O'), 1: ('Age', 'Y'), 2: ('Sex', 'F'), ...}

        Returns
        -------
        tuple
            A tuple (remapped_intrinsic, remapped_transition) where:
              - remapped_intrinsic is a dict mapping integer column index to utility value.
              - remapped_transition is a dict mapping (from_index, to_index) to utility value.

        Notes
        -----
        - The method performs case-insensitive matching by converting attribute names and values to lowercase.
        - If a key in a utility table does not have a corresponding entry in column_values, it is skipped.
        """
        # Invert column_values to map (attribute.lower(), value.lower()) -> column index.
        inv_map = {(attr.lower(), val.lower()): idx for idx, (attr, val) in column_values.items()}

        remapped_intrinsic = {}
        # Remap intrinsic utility table keys: ('Attribute', 'Value') -> utility
        for key, utility in self.intrinsic_utility_table.items():
            # Normalize key to lowercase
            attr, val = key
            lookup_key = (attr.lower(), val.lower())
            # Look up the corresponding column index; if not found, skip this key.
            if lookup_key in inv_map:
                col_index = inv_map[lookup_key]
                remapped_intrinsic[col_index] = utility
            # Else: optionally, one could log or warn about a missing mapping.

        remapped_transition = {}
        # Remap transition utility table keys: ('Attribute', from_value, to_value) -> utility
        for key, utility in self.transition_utility_table.items():
            attr, from_val, to_val = key
            lookup_from = (attr.lower(), from_val.lower())
            lookup_to = (attr.lower(), to_val.lower())
            # Only remap if both the from and to values exist in inv_map.
            if lookup_from in inv_map and lookup_to in inv_map:
                from_index = inv_map[lookup_from]
                to_index = inv_map[lookup_to]
                remapped_transition[(from_index, to_index)] = utility
            # Else: skip or log missing mapping.

        return remapped_intrinsic, remapped_transition
