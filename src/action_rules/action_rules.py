"""Main class ActionRules."""

import itertools
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from .candidates.candidate_generator import CandidateGenerator
from .output.output import Output
from .rules.rules import Rules

if TYPE_CHECKING:
    import cupy
    import numpy


class ActionRules:
    """
    A class used to generate action rules for a given dataset.

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
    rules : Rules, optional
        Stores the generated rules.
    output : Output, optional
        Stores the generated action rules.

    Methods
    -------
    fit(data, stable_attributes, flexible_attributes, target, undesired_state, desired_state)
        Generates action rules based on the provided dataset and parameters.
    get_bindings(data, stable_attributes, flexible_attributes, target)
        Binds attributes to corresponding columns in the dataset.
    get_stop_list(stable_items_binding, flexible_items_binding)
        Generates a stop list to prevent certain combinations of attributes.
    get_split_tables(data, target_items_binding, target)
        Splits the dataset into tables based on target item bindings.
    get_rules()
        Returns the generated action rules if available.
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

    def get_array_library(self, use_gpu: bool) -> Union['numpy', 'cupy']:
        """
        Return the appropriate DataFrame library (cuDF or pandas) based on the user's preference and availability.

        Parameters
        ----------
        use_gpu : bool
            Indicates whether to use GPU (cuDF) for data processing if available.

        Returns
        -------
        Union[cupy, numpy]
            The cuDF library if `use_gpu` is True and CuPy is available; otherwise, the Numpy library.

        Raises
        ------
        ImportError
            If `use_gpu` is True but cuDF is not available and pandas cannot be imported as fallback.

        Warnings
        --------
        UserWarning
            If `use_gpu` is True but cuDF is not available, a warning is issued indicating fallback to pandas.
        """
        if use_gpu:
            try:
                import cupy as np
            except ImportError:
                warnings.warn("CuPy is not available. Falling back to Numpy.")
                import numpy as np
        else:
            import numpy as np

        return np

    def df_to_array(self, df: pd.DataFrame, use_gpu: bool = False) -> tuple:
        """
        Convert a pandas DataFrame to a numpy array or a CuPy array.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to convert.
        use_gpu : bool, optional
            If True, the data will be converted to a GPU array using CuPy. Default is False.

        Returns
        -------
        tuple
            A tuple containing the transposed array and the DataFrame columns.

        Notes
        -----
        The data is converted to an unsigned 8-bit integer array (`np.uint8`). If `use_gpu` is True,
        the array is further converted to a CuPy array.
        """
        columns = df.columns
        data = df.to_numpy(dtype=np.uint8)
        if use_gpu:
            cp = self.get_array_library(use_gpu)
            data = cp.asarray(data, dtype=cp.uint8)
        return data.T, columns

    def one_hot_encode(
        self, data: pd.DataFrame, stable_attributes: list, flexible_attributes: list, target: str
    ) -> pd.DataFrame:
        """
        Perform one-hot encoding on the specified stable, flexible, and target attributes of the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the data to be encoded.
        stable_attributes : list
            List of stable attributes to be one-hot encoded.
        flexible_attributes : list
            List of flexible attributes to be one-hot encoded.
        target : str
            The target attribute to be one-hot encoded.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified attributes one-hot encoded.

        Notes
        -----
        The input data is first converted to string type to ensure consistent encoding. The stable attributes,
        flexible attributes, and target attribute are then one-hot encoded separately and concatenated into a
        single DataFrame.
        """
        data = data.astype(str)
        data_stable = pd.get_dummies(data[stable_attributes], sparse=False, prefix_sep='_<item_stable>_')
        data_flexible = pd.get_dummies(data[flexible_attributes], sparse=False, prefix_sep='_<item_flexible>_')
        data_target = pd.get_dummies(data[[target]], sparse=False, prefix_sep='_<item_target>_')
        data = pd.concat([data_stable, data_flexible, data_target], axis=1)
        return data

    def fit(
        self,
        data: pd.DataFrame,
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
        target_undesired_state: str,
        target_desired_state: str,
        use_gpu: bool = False,
    ):
        """
        Generate action rules based on the provided dataset and parameters.

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
        use_gpu : bool, optional
            Use GPU (cuDF) for data processing if available.
        """
        data = self.one_hot_encode(data, stable_attributes, flexible_attributes, target)
        data, columns = self.df_to_array(data, use_gpu)

        stable_items_binding, flexible_items_binding, target_items_binding = self.get_bindings(
            columns, stable_attributes, flexible_attributes, target
        )
        stop_list = self.get_stop_list(stable_items_binding, flexible_items_binding)
        frames = self.get_split_tables(data, target_items_binding, target, columns)
        undesired_state = columns.get_loc(target + '_<item_target>_' + str(target_undesired_state))
        desired_state = columns.get_loc(target + '_<item_target>_' + str(target_desired_state))

        stop_list_itemset = []  # type: list

        candidates_queue = [
            {
                'ar_prefix': tuple(),
                'itemset_prefix': tuple(),
                'stable_items_binding': stable_items_binding,
                'flexible_items_binding': flexible_items_binding,
                'undesired_mask': None,
                'desired_mask': None,
                'actionable_attributes': 0,
            }
        ]
        k = 0
        self.rules = Rules(undesired_state, desired_state, columns)
        candidate_generator = CandidateGenerator(
            frames,
            self.min_stable_attributes,
            self.min_flexible_attributes,
            self.min_undesired_support,
            self.min_desired_support,
            self.min_undesired_confidence,
            self.min_desired_confidence,
            undesired_state,
            desired_state,
            self.rules,
        )
        while len(candidates_queue) > 0:
            candidate = candidates_queue.pop(0)
            if len(candidate['ar_prefix']) > k:
                k += 1
                self.rules.prune_classification_rules(k, stop_list)
            new_candidates = candidate_generator.generate_candidates(
                **candidate,
                stop_list=stop_list,
                stop_list_itemset=stop_list_itemset,
                undesired_state=undesired_state,
                desired_state=desired_state,
                verbose=self.verbose,
            )
            candidates_queue += new_candidates
        self.rules.generate_action_rules()
        self.output = Output(self.rules.action_rules, target)

    def get_bindings(
        self,
        columns: pd.core.indexes.base.Index,
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
    ) -> tuple:
        """
        Bind attributes to corresponding columns in the dataset.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The dataset containing the attributes.
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
        """
        stable_items_binding = defaultdict(lambda: [])
        flexible_items_binding = defaultdict(lambda: [])
        target_items_binding = defaultdict(lambda: [])

        for i, col in enumerate(columns):
            is_continue = False
            # stable
            for attribute in stable_attributes:
                if col.startswith(attribute + '_<item_stable>_'):
                    stable_items_binding[attribute].append(i)
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # flexible
            for attribute in flexible_attributes:
                if col.startswith(attribute + '_<item_flexible>_'):
                    flexible_items_binding[attribute].append(i)
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # target
            if col.startswith(target + '_<item_target>_'):
                target_items_binding[target].append(i)
        return stable_items_binding, flexible_items_binding, target_items_binding

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
        """
        stop_list = []
        for items in stable_items_binding.values():
            for stop_couple in itertools.product(items, repeat=2):
                stop_list.append(tuple(stop_couple))
        for item in flexible_items_binding.keys():
            stop_list.append(tuple([item, item]))
        return stop_list

    def get_split_tables(
        self,
        data: Union['numpy.ndarray', 'cupy.ndarray'],
        target_items_binding: dict,
        target: str,
        columns: pd.core.indexes.base.Index,
    ) -> dict:
        """
        Split the dataset into tables based on target item bindings.

        Parameters
        ----------
        data : Union[numpy.ndarray, cupy.ndarray]
            The dataset to be split.
        target_items_binding : dict
            Dictionary containing bindings for target items.
        target : str
            The target attribute.

        Returns
        -------
        dict
            A dictionary containing the split tables.
        """
        frames = {}
        for item in target_items_binding[target]:
            mask = data[item] == 1
            frames[item] = data[:, mask]
        return frames

    def get_rules(self) -> Optional[Output]:
        """
        Return the generated action rules if available.

        Returns
        -------
        Optional[Output]
            The generated action rules, or None if no rules have been generated.
        """
        if self.output is None:
            return None
        else:
            return self.output
