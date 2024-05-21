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

    def __init__(self, N, beta, truncate=4):

        self.N = N
        self.N_curr = N
        self.beta = beta
        self.truncate = truncate

        self.transfer_tensor = None
        self.Z = 1
        # self.tensor_network = None

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
                        transfer_tensor[t, r, b, l] = np.exp(-self.beta * spins)

        self.transfer_tensor = transfer_tensor

        # tensor_network = TensorNetwork(
        #     [transfer_tensor] * self.N**2,
        #     [(i, j) for i in range(self.N) for j in range(self.N)],
        # )

        # self.tensor_network = tensor_network

        return transfer_tensor

    def svd(self, tensor, orientation="left", truncate=None):
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

        shape = tensor.shape

        # reshape tensor for SVD
        if orientation == "left":
            tensor = tensor.transpose(1, 2, 3, 0).reshape(
                shape[1] * shape[2], shape[3] * shape[0]
            )
        elif orientation == "right":
            tensor = tensor.reshape(shape[0] * shape[1], shape[2] * shape[3])
        else:
            raise ValueError("Invalid orientation.")

        # perform SVD
        U, S, V = np.linalg.svd(tensor, full_matrices=False)

        # truncate singular values
        if truncate is not None:
            U = U[:, :truncate]
            S = S[:truncate]
            V = V[:truncate, :]

        # contract singular values back into U and V
        U = U @ np.diag(np.sqrt(S))
        V = np.diag(np.sqrt(S)) @ V

        # reshape tensors
        if orientation == "left":
            U = U.reshape((shape[3], shape[0], -1))  # dav
            V = V.reshape((-1, shape[1], shape[2]))  # wbc
        elif orientation == "right":
            U = U.reshape((shape[2], shape[3], -1))  # ghx
            V = V.reshape((-1, shape[0], shape[1]))  # yef
        else:
            raise ValueError("Invalid orientation.")

        return U, V

    def update(self):
        """"""

        # decompose transfer tensor with SVD
        U_l, V_l = self.svd(
            self.transfer_tensor, orientation="left", truncate=self.truncate
        )
        U_r, V_r = self.svd(
            self.transfer_tensor, orientation="right", truncate=self.truncate
        )

        # contract the plaquette
        ten_1 = np.einsum("wic,gix->wxgc", V_l, U_r)
        ten_2 = np.einsum("yei,iav->eavy", V_r, U_l)
        ten_3 = np.einsum("wxij,jivy->wxvy", ten_1, ten_2)

        # ten_1 = np.tensordot(V_l, U_r, axes=(1, 1))
        # ten_2 = np.tensordot(V_r, U_l, axes=(2, 1))
        # ten_3 = np.tensordot(ten_1, ten_2, axes=([2, 3], [1, 2]))

        # compute the partition function
        trace = np.trace(np.trace(ten_3, axis1=0, axis2=2), axis1=0, axis2=1)

        self.transfer_tensor = np.copy(ten_3 / trace)
        self.Z = self.Z * trace ** (1 / 2**self.N_curr)

        return self.Z

    def solve(self):
        """
        Compute the partition function.
        """

        self.initialize()

        for n in range(1, self.N + 1):
            Z = self.update()
            self.N_curr = n

        return np.log2(Z)


class Tensor:

    def __init__(self, array, inds=None):
        self.array = array
        self.inds = inds


class TensorNetwork:

    def __init__(self, tensors, inds):
        self.tensors = tensors
        self.inds = inds
