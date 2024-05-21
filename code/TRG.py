import numpy as np


class TRG:
    """
    The TRG class implements the tensor renormalization group algorithm for the 2D Ising model.

    Parameters
    ----------
    N : int
        The size of the lattice.
    beta : float
        The inverse temperature.
    """

    def __init__(self, N, beta):

        self.N = N
        self.beta = beta
        self.tensor_network = None

    def svd(self, tensor, inds, orientation="left"):
        """
        Perform the singular value decomposition of a tensor along the specified indices. The orientation determines which indices to group together.

        Parameters
        ----------
        tensor : Tensor
            The tensor to perform the SVD on.
        inds : list of int
            The indices to group together.
        orientation : str
            The orientation of the indices to group together. Either "left" or "right".
        """

        ...

        return

    def initialize(self):
        """
        Initialize the initial tensor network.
        """

        transfer_tensor = np.zeros((2, 2, 2, 2))

        for t in [-1, 1]:
            for r in [-1, 1]:
                for b in [-1, 1]:
                    for l in [-1, 1]:
                        spins = t * r + r * b + b * l + l * t

                        transfer_tensor[t, r, b, l] = np.exp(-self.beta * spins)

        transfer_tensor = Tensor(transfer_tensor, ["t", "r", "b", "l"])

        tensor_network = TensorNetwork(
            [transfer_tensor] * self.N**2, np.arange(self.N**2)
        )

        self.tensor_network = tensor_network

    def update(self):
        """
        Do one update step.
        """

        ...

        return

    def solve(self):
        """
        Compute the partition function.
        """

        ...

        return


class Tensor:

    def __init__(self, array, inds=None):
        self.array = array
        self.inds = inds


class TensorNetwork:

    def __init__(self, tensors, inds):
        self.tensors = tensors
        self.inds = inds
