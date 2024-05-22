import numpy as np
from scipy.linalg import eig

from .TRG import TRG


class GiltTNR(TRG):
    """
    The GiltTNR implements the Gilt-TNR algorithm for the 2D Ising model (https://arxiv.org/abs/1709.07460).

    Parameters
    ----------
    N : int
        The size of the lattice.
    beta : float
        The inverse temperature.
    truncate : float or int
        The number of singular values to keep in the SVD. If an integer, the number of singular values to keep. If a float, the fraction of singular values to keep.
    """

    def __init__(self, N, beta, truncate=0.25):

        super().__init__(N, beta, truncate)

        # data to update
        self.blue_transfer_tensor = None
        self.red_transfer_tensor = None
        self.gilt_tensor = None

    def create_gilt_tensor(self, S, U, error=1e-3):
        """
        Initialize the initial gilt tensor.
        """
        t = np.einsum("abc,c", U, S).flatten()

        t_prime = t * (S**2 / (error**2 + S**2))

        gilt_tensor = np.einsum("abc,c", U, t_prime)

        self.gilt_tensor = gilt_tensor

        return gilt_tensor

    def optimize_gilt_tensor(self):
        """
        Optimize the gilt tensor.
        """

        ...

        return

    def svd_gilt_tensor(self, tensor, truncate):
        """
        Perform the SVD on the gilt tensor.
        """

        # perform the SVD
        U, S, V = np.linalg.svd(tensor, full_matrices=False)

        # truncate the SVD
        if isinstance(truncate, int):
            U = U[:, :truncate]
            S = S[:truncate]
            V = V[:truncate, :]
        elif isinstance(truncate, float):
            U = U[:, : int(truncate * len(S))]
            S = S[: int(truncate * len(S))]
            V = V[: int(truncate * len(S)), :]
        elif truncate is None:
            pass
        else:
            raise ValueError("The truncate parameter must be an integer or a float.")

        # contract singular values back into U and V
        U = U @ np.diag(np.sqrt(S))
        V = np.diag(np.sqrt(S)) @ V

        return U, V

    def apply_gilt_tensor(self, U, V, bond):
        """"""

        if bond == "t":
            self.blue_transfer_tensor = np.einsum(
                "axcd,ex->aecd", self.blue_transfer_tensor, U
            )
            self.red_transfer_tensor = np.einsum(
                "xf,abcx->abcf", V, self.red_transfer_tensor
            )
        elif bond == "l":
            self.red_transfer_tensor = np.einsum(
                "xbcd,ex->ebcd", self.red_transfer_tensor, U
            )
            self.blue_transfer_tensor = np.einsum(
                "xf,abxd->abfd", V, self.blue_transfer_tensor
            )
        elif bond == "b":
            self.red_transfer_tensor = np.einsum(
                "axcd,xf->afcd", self.red_transfer_tensor, U
            )
            self.blue_transfer_tensor = np.einsum(
                "ex,abcx->abce", V, self.blue_transfer_tensor
            )
        elif bond == "r":
            self.blue_transfer_tensor = np.einsum(
                "xbcd,ex->ebcd", self.blue_transfer_tensor, U
            )
            self.red_transfer_tensor = np.einsum(
                "xf,abxd->abfd", V, self.red_transfer_tensor
            )

    def update_gilt_tensor(self):
        """"""

        for bond in ["t", "l", "b", "r"]:
            S, U = self.compute_env_spectrum(bond)

            self.create_gilt_tensor(S, U)

            U, V = self.svd_gilt_tensor(self.gilt_tensor, 5)

            self.apply_gilt_tensor(U, V, bond)

    def compute_env_spectrum(self, bond):
        """
        Compute the environment spectrum for a plaquette. Since only U and S are needed, do the eigenvalues instead of the SVD ($EE^\dagger = US^2U\dagger$). The bond is represented by a string of the form "t", "r", "b", or "l", which corresponds to the top, right, bottom, or left bond of the plaquette.

        Parameters
        ----------
        bond : str
            The bond to compute the environment spectrum for.
        """

        shape_blue = self.blue_transfer_tensor.shape
        shape_red = self.red_transfer_tensor.shape

        # regroup the transfer tensors
        TL = self.blue_transfer_tensor.transpose(1, 2, 3, 0).reshape(
            shape_blue[1] * shape_blue[2], shape_blue[3], shape_blue[0]
        )  # abc
        LB = self.red_transfer_tensor.reshape(
            shape_red[0], shape_red[1], shape_red[2] * shape_red[3]
        )  # def
        BR = self.blue_transfer_tensor.reshape(
            shape_blue[0], shape_blue[1] * shape_blue[2], shape_blue[3]
        )  # ghi
        RT = self.red_transfer_tensor.reshape(
            shape_red[0] * shape_red[1], shape_red[2], shape_red[3]
        )  # jkl

        # contract the environment tensor
        if bond == "t":
            tensor = np.einsum("abx,xyf,zhy,jzl->bljhfa", TL, LB, BR, RT)
            shape = tensor.shape
            tensor = tensor.reshape(shape[0] * shape[1], -1)
        elif bond == "l":
            tensor = np.einsum("dxf,yhx,jyz,azc->dcajhf", LB, BR, RT, TL)
            shape = tensor.shape
            tensor = tensor.reshape(shape[0] * shape[1], -1)
        elif bond == "b":
            tensor = np.einsum("xhi,jxy,ayz,zef->iefajh", BR, RT, TL, LB)
            shape = tensor.shape
            tensor = tensor.reshape(shape[0] * shape[1], -1)
        elif bond == "r":
            tensor = np.einsum("jkx,axy,yzf,ghz->kghfaj", RT, TL, LB, BR)
            shape = tensor.shape
            tensor = tensor.reshape(shape[0] * shape[1], -1)

        # compute the environment spectrum
        S, U = np.linalg.eig(tensor @ tensor.T)

        U = U.reshape(shape[0], shape[1], -1)

        S = np.sqrt(np.abs(S))

        return S, U

    def update(self, n):
        """
        Update the gilt tensor.
        """

        self.blue_transfer_tensor = self.transfer_tensor
        self.red_transfer_tensor = self.transfer_tensor
        print(self.red_transfer_tensor.shape)
        self.update_gilt_tensor()

        # decompose transfer tensor with SVD
        U_blue_l, V_blue_l = self.svd_transfer_tensor(
            self.blue_transfer_tensor, orientation="left"
        )
        # U_blue_r, V_blue_r = self.svd_transfer_tensor(
        #     self.blue_transfer_tensor, orientation="right"
        # )

        # U_red_l, V_red_l = self.svd_transfer_tensor(
        #     self.red_transfer_tensor, orientation="left"
        # )
        U_red_r, V_red_r = self.svd_transfer_tensor(
            self.red_transfer_tensor, orientation="right"
        )

        # contract the plaquette
        ten_1 = np.einsum("wic,gix->wxgc", V_blue_l, U_red_r)
        ten_2 = np.einsum("yei,iav->eavy", V_red_r, U_blue_l)
        ten_3 = np.einsum("wxij,jivy->wxvy", ten_1, ten_2)

        # compute the partition function
        trace = np.trace(np.trace(ten_3, axis1=0, axis2=2), axis1=0, axis2=1)

        self.transfer_tensor = np.copy(ten_3 / trace)

        self.Z = self.Z * trace ** (1 / 2 ** (n + 1))

        return self.Z
