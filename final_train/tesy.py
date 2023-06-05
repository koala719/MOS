from collections import namedtuple
import numpy as np

num_cells = 5
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
fixed_cell = "0 1 1 1 1 2 1 2 3 2 2 1 4 2 2 4 0 4 0 0 0 0 1 4 2 0 0 2 0 1 0 1 4 1 1 0 1 2 2 0"
fixed_cell = np.array([int(x) for x in fixed_cell.split(" ") if x])
normal_cell = fixed_cell[:4 * num_cells]
reduce_cell = fixed_cell[4 * num_cells:]

def arc(cell):
    cell_op = []
    cell_id = []
    for i, index in enumerate(cell):
        if (i%2) == 0:
            cell_id.append(index)
        else:
            cell_op.append(index)
    full_id = [0, 1, 2, 3, 4, 5, 6]
    no_used = [i for i in full_id if i not in cell_id]
    arc = []
    for i in range(len(cell_id)):
        arc.append([str(cell_op[i]), cell_id[i]])
    return arc, no_used
ENAS = Genotype(normal=arc(normal_cell)[0], normal_concat = arc(normal_cell)[1], reduce =arc(reduce_cell)[0], reduce_concat = arc(reduce_cell)[1])
print(ENAS)