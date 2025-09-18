"""Utilities for sensitivity analysis.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch


def build_single_cov_matrix(std, intra_cor=None):
    """Build a single covariance matrix from std. and correlation coefficients.

    The method starts with building a correlation matrix Ck as

        [  1 intra_cor[0]   intra_cor[1] ...      intra_cor[N-2]]
    C = [...            1 intra_cor[N-1] ...   intra_cor[2N - 2]].
        [...          ...            ... ...                 ...]
        [...          ...            ... ... intra_cor[N(N-1)/2]]

    It then builds the covariance matrix as

    S = diag(std) * C * diag(std).

    Args:
        std (torch.Tensor): (N,) tensor holding standard deviations of basic
        parameter for which we wish to compute a covariance matrix.

        intra_cor (torch.Tensor, optional): (N*(N-1)/2,) tensor holding
        correlation coefficients between elements of basic parameter for which
        we wish to compute a covariance matrix. If None, the correlation is
        assumed to be zero. Defaults to None

    Returns:
        torch.Tensor: (N, N) torch tensor corresponding to covariance matrix.
    """
    N = len(std)
    counter = 0
    if intra_cor is not None:
        C = torch.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                C[i, j] = intra_cor[counter]
                counter += 1

        C = C + C.T + torch.diag(torch.ones(N))
    else:
        C = torch.eye(N)

    D = torch.diag(std)

    return D @ C @ D


def build_cross_cov_matrix(cov, inter_cor=None):
    """Build a valid cross-covariance from a covariance matrix.

    Given a covariance matrix S common to two random variables X and Y, this
    function builds the larger covariance matrix.

    S_ = [S,       rho * S]
         [rho * S,       S]

    Args:
        cov (torch.Tensor): (N, N) tensor holding the covariance matrix of the
        parameters for which we wish to compute a cross-covariance matrix.

        inter_cor (float, optional): cross-correlation coefficient. If None, it
        is assumed to be zero. Defaults to None.
    """

    if inter_cor is None or inter_cor == 0.0:
        return torch.zeros_like(cov)
    else:
        return inter_cor * cov


def is_valid_covariance(cov, semi_definite=True, tol=100*core.TEPS):
    """Test whether input is a valid covariance matrix.

    Args:
        cov (torch.Tensor): input matrix to be tested.

        semi_definite (bool, optional): if True, test for positive
        semi-definiteness; otherwise, test for positive definiteness. Defaults
        to True.
    """

    # Has tensor rank 2?
    if cov.dim() != 2:
        return False
    # Is square?
    elif cov.shape[0] != cov.shape[1]:
        return False
    # Is symmetric?
    elif not torch.allclose(cov, cov.T, rtol=tol, atol=tol):
        return False

    # Is positive (semi-)definite?
    eigvals, _ = torch.linalg.eigh(cov)
    # Normalize the eigenvalues.
    eigvals = eigvals / max(eigvals[-1].item(), 1.0)
    if semi_definite:
        op = torch.lt
        # Test for positive semi-definiteness needs a tolerance.
        zeros = -tol * torch.ones_like(eigvals)
    else:
        op = torch.le
        # Test for strict positive definiteness does not need a tolerance.
        zeros = torch.zeros_like(eigvals)

    if torch.any(op(eigvals, zeros)):
        return False
    else:
        return True


def stack_covariances(first_cov, second_cov):
    """Stack two covariance matrices as diagonal blocks.

    Args:
        first_cov (torch.Tensor): upper-left block.

        second_cov (torch.Tensor): lower-right block.

    Returns:
        torch.Tensor: covariance matrix with each input occupying a block.
    """
    M, N = first_cov.shape
    I, J = second_cov.shape
    return \
        torch.vstack(
            (
                torch.hstack((first_cov, torch.zeros((M, J)))),
                torch.hstack((torch.zeros((I, N)), second_cov))
            ),
        )
