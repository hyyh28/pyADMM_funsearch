
def priority(bin, item):
    """
    Returns priority with which we want to add item to each bin.

    Args:
        bin : Total available CPU resources.
        item : Item need to be placed.

    Returns:
        int: The total score for current bin.

    You should consider the following factor which may be helpful to optimize the result.The factor is "flavor_types".
    "flavor_types":It is a dict that indicate the potential flavor types for future virtual machine requests.
    """

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
