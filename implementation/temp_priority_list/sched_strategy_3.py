
def priority(bin, item):
    """
    Returns priority with which we want to add item to each bin.

    Args:
        bin (int): Total available CPU resources.
        item (int): Item need to be placed.

    Returns:
        int: The total score for current bin.

    You should consider the following factor which may be helpful to optimize the result.The factor is "initial_size_pyhsical_machine".
    "initial_size_pyhsical_machine":It is a two dimension vector that indicates the initial CPU and memory resources of a physical machine.
    
    """
    initial_size_pyhsical_machine = {'cpu':128,'mem':160}

    score = -( bin[0] - item[0] ) 
    return score
