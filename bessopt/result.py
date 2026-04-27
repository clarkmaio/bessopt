

import polars as pl
import numpy as np

class OptimisationResult:
    """
    Class to store the results of an optimisation in a nice format easy to handle.
    """

    def __init__(self):
        self._data = pl.DataFrame()



    def __repr__(self):
        pass


    def __html_repr__(self):
        '''
        Marimo rapresentation
        '''
        pass