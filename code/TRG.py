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
    truncate : int
        The number of singular values to keep in the SVD.

    Attributes
    ----------
    N : int
        The size of the lattice.
    N_curr : int
        The current size of the lattice.
    beta : float
        The inverse temperature.
    truncate : int or float
        The number of singular values to keep in the SVD. If an integer, the number of singular values to keep. If a float, the fraction of singular values to keep.
    transfer_tensor : np.ndarray
        The transfer tensor.
    Z : float
        The partition function.
    """

    def __init__(self, N, beta, truncate=10):

        self.N = N
        self.N_curr = N
        self.beta = beta
        self.truncate = truncate

        self.transfer_tensor = None
        self.Z = 1

    def initialize(self):
        """
        Initialize the initial transfer tensor.
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

        return transfer_tensor

    def svd(self, tensor, orientation="left"):
        """
        Perform the singular value decomposition of a tensor. The orientation determines which indices to group together.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to perform the SVD on.
        orientation : str
            The orientation of the indices to group together. Either "left" or "right".
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
        if isinstance(self.truncate, int):
            U = U[:, : self.truncate]
            S = S[: self.truncate]
            V = V[: self.truncate, :]
        elif isinstance(self.truncate, float) and 0 <= self.truncate <= 1:
            U = U[:, : int(self.truncate * len(S))]
            S = S[: int(self.truncate * len(S))]
            V = V[: int(self.truncate * len(S)), :]
        elif isinstance(self.truncate, None):
            pass
        else:
            raise ValueError("Invalid truncate value.")

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

    def update(self, n):
        """
        Update the transfer tensor, i.e. one step of the TRG algorithm.
        """

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

        # compute the partition function
        trace = np.trace(np.trace(ten_3, axis1=0, axis2=2), axis1=0, axis2=1)

        self.transfer_tensor = np.copy(ten_3 / trace)
        self.Z = self.Z * trace ** (1 / 2**n)

        return self.Z

    def solve(self):
        """
        Compute the partition function.
        """

        self.initialize()

        for n in range(1, self.N + 1):
            Z = self.update(n)

        return np.log2(Z)
