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

    def __init__(self, N, beta, truncate=2):

        self.N = N
        self.N_curr = N
        self.beta = beta
        self.truncate = truncate

        self.tensor_network = None

    def initialize(self):
        """
        Initialize the initial tensor network.
        """

        transfer_tensor = np.zeros((2, 2, 2, 2))

        for t in [0, 1]:
            for r in [0, 1]:
                for b in [0, 1]:
                    for l in [0, 1]:
                        spins = (
                            (2 * t - 1) * (2 * r - 1)
                            + (2 * r - 1) * (2 * b - 1)
                            + (2 * b - 1) * (2 * l - 1)
                            + (2 * l - 1) * (2 * t - 1)
                        )
                        transfer_tensor[t, r, b, l] = np.exp(self.beta * spins)

        transfer_tensor = transfer_tensor

        tensor_network = TensorNetwork(
            [transfer_tensor] * self.N**2,
            [(i, j) for i in range(self.N) for j in range(self.N)],
        )

        self.tensor_network = tensor_network

        return tensor_network

    def svd(self, tensor, orientation="left", truncate=0):
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
            The number of singular values to remove.
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
        U = U[:, : 4 - truncate]
        S = S[: 4 - truncate]
        V = V[: 4 - truncate, :]

        # contract singular values back into U and V
        U = U @ np.diag(np.sqrt(S))
        V = np.diag(np.sqrt(S)) @ V

        # reshape tensors
        if orientation == "left":
            U = U.reshape((2, 2, -1))
            V = V.reshape((-1, 2, 2))
        elif orientation == "right":
            U = U.reshape((2, 2, -1))
            V = V.reshape((-1, 2, 2))
        else:
            raise ValueError("Invalid orientation.")

        return U, V

    def contract_plaquette(self, ten_a, ten_b, ten_c, ten_d):

        ten_1 = np.tensordot(ten_a[1], ten_b[1], axes=(1, 1))

        ten_2 = np.tensordot(ten_c[1], ten_d[1], axes=(2, 1))

        ten_3 = np.tensordot(ten_1, ten_2, axes=([2, 3], [1, 2]))

        return ten_3

    def update(self):
        """
        Do one update step.
        """

        tn = []

        # compute svd
        for num_tensor, tensor in enumerate(self.tensor_network.tensors):

            if num_tensor % 2 == 0:
                U, V = self.svd(tensor, orientation="left", truncate=self.truncate)
            else:
                U, V = self.svd(tensor, orientation="right", truncate=self.truncate)

            tn.append([U, V])

        # EXAMPLE
        num_tensor = 0
        tensor = self.contract_plaquette(
            tn[num_tensor],
            tn[num_tensor + 1],
            tn[num_tensor + self.N],
            tn[num_tensor + self.N - 1],
        )

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
