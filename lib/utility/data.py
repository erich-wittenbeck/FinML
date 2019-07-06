
def create_diagonal_mask(*dims, invert=False):
    show, hide = 0, 1
    if invert:
        show, hide = 1, 0

    mask = [[show if j == i else hide for j in range(dims[1])] for i in range(dims[0])]

    return mask