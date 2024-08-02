
def priority(bin, item):
    """
    Returns priority with which we want to add item to each bin.

    Args:
        bin (int): Total available CPU resources.
        item (int): Item need to be placed.

    Returns:
        int: The total score for current bin.

    You should consider the following factor which may be helpful to optimize the result.The factor is "capacity_dict".
    "capacity_dict":It is a dict that indicates how many VMs can be placed for the current PM for each flavor type.
    """
    
    cpu = bin[0]
    mem = bin[1]
    capacity_dict = [
        {'id': 0, 'cpu': 2, 'mem': 8, 'Quantity_placed':min(cpu // 2, mem // 8)},
        {'id': 1, 'cpu': 8, 'mem': 32,'Quantity_placed':min(cpu // 8, mem // 32)},
        {'id': 2, 'cpu': 16, 'mem': 32, 'Quantity_placed':min(cpu // 16, mem // 32)},
        {'id': 3, 'cpu': 16, 'mem': 1,'Quantity_placed':min(cpu // 16, mem // 1)},
        {'id': 4, 'cpu': 6, 'mem': 4, 'Quantity_placed':min(cpu // 6, mem // 4)},
        {'id': 5, 'cpu': 2, 'mem': 16,'Quantity_placed':min(cpu // 2, mem // 16)},
        {'id': 6, 'cpu': 96, 'mem': 64,'Quantity_placed':min(cpu // 96, mem // 64)},
        {'id': 7, 'cpu': 4, 'mem': 16, 'Quantity_placed':min(cpu // 4, mem // 16)},
        {'id': 8, 'cpu': 8, 'mem': 16, 'Quantity_placed':min(cpu // 8, mem // 16)},
        {'id': 9, 'cpu': 16, 'mem': 64, 'Quantity_placed':min(cpu // 16, mem // 64)},
        {'id': 10, 'cpu': 48, 'mem': 32, 'Quantity_placed':min(cpu // 48, mem // 32)},
        {'id': 11, 'cpu': 12, 'mem': 8, 'Quantity_placed':min(cpu // 12, mem // 8)},
        {'id': 12, 'cpu': 24, 'mem': 16, 'Quantity_placed':min(cpu // 24, mem // 16)},
        {'id': 13, 'cpu': 36, 'mem': 24, 'Quantity_placed':min(cpu // 36, mem // 24)},
        {'id': 14, 'cpu': 72, 'mem': 48, 'Quantity_placed':min(cpu // 72, mem // 48)},
        {'id': 15, 'cpu': 2, 'mem': 32, 'Quantity_placed':min(cpu // 2, mem // 32)},
        {'id': 16, 'cpu': 4, 'mem': 8,  'Quantity_placed':min(cpu // 4, mem // 8)}
    ]

    score = -( bin[0] - item[0] ) 
    return score
