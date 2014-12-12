from numpy import exp, ones

DEFAULT_FACTOR = 1.0

SHAPES = {
        ((1,1,1),
         (1,1,1),
         (1,1,1)) : -3*DEFAULT_FACTOR,

        ((0,0,0),
         (1,1,1),
         (1,1,1)) : -2*DEFAULT_FACTOR,

        ((1,1,1),
         (1,1,1),
         (0,0,0)) : -2*DEFAULT_FACTOR,

        ((0,1,1),
         (0,1,1),
         (0,1,1)) : -2*DEFAULT_FACTOR,

        ((1,1,0),
         (1,1,0),
         (1,1,0)) : -2*DEFAULT_FACTOR,

        ((0,1,1),
         (1,1,1),
         (1,1,1)) : -DEFAULT_FACTOR,

        ((0,0,1),
         (1,1,1),
         (1,1,1)) : -DEFAULT_FACTOR,

        ((0,1,1),
         (0,1,1),
         (1,1,1)) : -DEFAULT_FACTOR,

        ((0,0,1),
         (0,1,1),
         (1,1,1)) : -DEFAULT_FACTOR,

        ((0,0,0),
         (0,1,1),
         (1,1,1)) : -DEFAULT_FACTOR,

        ((0,0,1),
         (0,1,1),
         (0,1,1)) : -DEFAULT_FACTOR,

        ((0,0,0),
         (0,1,1),
         (0,1,1)) : -DEFAULT_FACTOR,

        ((0,1,1),
         (1,1,1),
         (1,1,0)) : 0.,

        ((0,0,0),
         (0,1,0),
         (0,0,0)) : 10000*DEFAULT_FACTOR,
}

def rot(shape, i):
    """
    Rotate a 3x3 tuple clockwise i times.
    """
    if i < 1:
        raise ValueError("rot i must be >= 1")
    new_shape = ((shape[2][0], shape[1][0], shape[0][0]),
                 (shape[2][1], shape[1][1], shape[0][1]),
                 (shape[2][2], shape[1][2], shape[0][2]))
    if i > 1:
        return rot(new_shape, i-1)
    else:
        return new_shape

# Add symmetric shapes
for shape, factor in SHAPES.items():
    for i in xrange(1,4):
        SHAPES[rot(shape, i)] = factor

def clique_to_shape(clique):
    """
    Cast the clique as a 3x3 tuple of cluster identities.
    """
    cluster = clique[1][1]
    return tuple((tuple((int(r == cluster) for r in row)) for row in clique))


def phi(clique, p=-DEFAULT_FACTOR, q=DEFAULT_FACTOR,
        default_shape=DEFAULT_FACTOR, extra_shapes=SHAPES):
    """
    Compute log factor function for a 3x3 clique around y_i

    Params
    ------
    clique : 3x3 ndarray
        The cluster assignments for the given clique
    p : float
        Factor for identity
    q : float
        Factor for difference
    default_shape : float
        Factor for default shape
    extra_shapes: dict
        Maps special shapes to their factors
    """
    shape = clique_to_shape(clique)
    if shape in extra_shapes:
        return -extra_shapes[shape]
    else:
        return -default_shape
