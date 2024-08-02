
def priority(bin, item):
    """
    Returns priority with which we want to add item to each bin.

    Args:
        bin (int): Total available CPU resources.
        item (int): Item need to be placed.

    Returns:
        int: The total score for current bin.

    You should consider the following factor which may be helpful to optimize the result.The factors are "initial_size_pyhsical_machine","current_size_pyhsical_machine","capacity_dict" and "flavor_types".
    "initial_size_pyhsical_machine":It is a two dimension vector that indicates the initial CPU and memory resources of a physical machine.
    "current_size_pyhsical_machine":It is a two dimension vector that indicates the current remained CPU and memory resources of a physical machine.
    "capacity_dict":It is a dict that indicates how many VMs can be placed for the current PM for each flavor type.
    "flavor_types":It is a dict that indicate the potential flavor types for future virtual machine requests.
    """
    cpu = bin[0]
    mem = bin[1]
    initial_size_pyhsical_machine = {'cpu':128,'mem':160}
    current_size_pyhsical_machine = {'cpu':cpu,'mem':mem}
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

    flavor_types = [
        {'id': 0, 'cpu': 2, 'mem': 8, 'isdouble':0},
        {'id': 1, 'cpu': 8, 'mem': 32,'isdouble':0},
        {'id': 2, 'cpu': 192, 'mem': 128,'isdouble':1},
        {'id': 3, 'cpu': 16, 'mem': 32,'isdouble':0},
        {'id': 4, 'cpu': 16, 'mem': 1,'isdouble':0},
        {'id': 5, 'cpu': 6, 'mem': 4, 'isdouble':0},
        {'id': 6, 'cpu': 2, 'mem': 16,'isdouble':0},
        {'id': 7, 'cpu': 96, 'mem': 64,'isdouble':0},
        {'id': 8, 'cpu': 4, 'mem': 16,'isdouble':0},
        {'id': 9, 'cpu': 8, 'mem': 16, 'isdouble':0},
        {'id': 10, 'cpu': 16, 'mem': 64, 'isdouble':0},
        {'id': 11, 'cpu': 48, 'mem': 32, 'isdouble':0},
        {'id': 12, 'cpu': 12, 'mem': 8, 'isdouble':0},
        {'id': 13, 'cpu': 24, 'mem': 16, 'isdouble':0},
        {'id': 14, 'cpu': 36, 'mem': 24, 'isdouble':0},
        {'id': 15, 'cpu': 72, 'mem': 48,'isdouble':0},
        {'id': 16, 'cpu': 144, 'mem': 96,'isdouble':1},
        {'id': 17, 'cpu': 2, 'mem': 32,'isdouble':0},
        {'id': 18, 'cpu': 4, 'mem': 8, 'isdouble':0},
        {'id': 19, 'cpu': 138, 'mem': 92,'isdouble':1}
    ]
    score = -( bin[0] - item[0] ) 
    return score
