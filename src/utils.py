import numpy as np
import torch

def get_receptive_field_of_output(list_all_layers:list)->tuple[int, int]:
    """
    Get the offset in width and height (only for top and left) from the output to have a whole receptive field

    Args:
        list_all_layers (list): list of all convolutions in the network

    Returns:
        tuple[int, int]: (offset left, offset top)
    """

    
    return 0