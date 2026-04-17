


from dataclasses import dataclass
from typing import Callable



@dataclass
class Utility(object):
    """
    Class to wrap cost utility function
    """

    function: Callable
    breakpoints: list
    
