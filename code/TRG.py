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

    def svd(self, tensor, inds, orientation="left", truncate=None):
        """
        Perform the singular value decomposition of a tensor along the specified indices. The orientation determines which indices to group together.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to perform the SVD on.
        inds : list of int
            The indices to group together.
        orientation : str
            The orientation of the indices to group together. Either "left" or "right".
        truncate : int
            The number of singular values to keep.
        """

        # reshape tensor for SVD
        if orientation == "left":
            tensor = tensor.transpose(1, 2, 3, 0).reshape(4, 4)
        elif orientation == "right":
            tensor = tensor.reshape(4, 4)
        else:
            raise ValueError("Invalid orientation.")

        # perform SVD
        U, S, V = np.linalg.svd(tensor)

        # truncate singular values
        U = U[:, :truncate]
        S = S[:truncate]
        V = V[:truncate, :]

        # contract singular values back into V
        W = np.tensordot(U, np.diag(S), (1, 0))

        # reshape tensors
        if orientation == "left":
            U = U.reshape((2, 2, truncate))
            W = W.reshape((2, 2, truncate))
        elif orientation == "right":
            V = V.reshape((truncate, 2, 2))
            W = W.reshape((truncate, 2, 2))
        else:
            raise ValueError("Invalid orientation.")

        return U, S, V

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

        transfer_tensor = transfer_tensor

        tensor_network = TensorNetwork(
            [transfer_tensor] * self.N**2, np.arange(self.N**2)
        )

        self.tensor_network = tensor_network

        return tensor_network

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
