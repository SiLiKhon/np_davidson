# https://gqcg-res.github.io/knowdes/the-davidson-diagonalization-method.html

from typing import Callable, Optional, Tuple
import numpy as np
import numpy.typing as npt


def _get_complement(trial: npt.NDArray, subspace: npt.NDArray) -> npt.NDArray:
    assert trial.ndim == 1
    assert subspace.ndim == 2
    assert subspace.shape[0] == trial.shape[0]

    vt = (subspace.T @ trial)
    return trial - subspace @ vt

def _normalize(vec: npt.NDArray):
    assert vec.ndim == 1
    return vec / np.linalg.norm(vec)

def _davidson_step(
    matmul: Callable,
    diag: npt.NDArray,
    trial: npt.NDArray,
    subspace: Optional[npt.NDArray] = None,
    subspace_A: Optional[npt.NDArray] = None,
    M: Optional[npt.NDArray] = None,
    thr: float = 1e-8,
) -> Tuple[
    Tuple[float, npt.NDArray],
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
]:
    if subspace is None:
        assert subspace_A is None
        assert M is None
    
        subspace = np.empty(shape=(trial.shape[0], 0), dtype=trial.dtype)
        subspace_A = np.empty(shape=(trial.shape[0], 0), dtype=trial.dtype)
        M = np.empty(shape=(0, 0), dtype=trial.dtype)

    assert diag.shape == trial.shape
    assert trial.ndim == 1
    assert subspace.ndim == 2
    assert subspace.shape[0] == trial.shape[0]
    assert subspace.shape == subspace_A.shape
    assert M.ndim == 2
    assert M.shape[0] == M.shape[1]
    assert subspace.shape[1] == M.shape[0]

    vk = _normalize(_get_complement(trial, subspace))
    vk_A = matmul(vk)

    subspace = np.c_[subspace, vk]
    subspace_A = np.c_[subspace_A, vk_A]

    mk = subspace.T @ vk_A

    M = np.c_[np.c_[M, mk[:-1]].T, mk]
    theta, s = np.linalg.eigh(M)
    theta = theta[0]
    s = s[:, 0]

    u = subspace @ s
    u_A = subspace_A @ s

    r = u_A - theta * u
    scale = theta - diag
    mask = np.abs(scale) < thr
    t = np.where(mask, 0, r / np.where(mask, 1, scale))

    return (
        (theta, u),
        (t, subspace, subspace_A, M),
    )

def smallest_eig(
    matmul: Callable,
    diag: npt.NDArray,
    trial: Optional[npt.NDArray] = None,
    max_steps: int = 15,
    convergence_threshold: float = 1e-7,
    verbose: bool = False,
) -> Tuple[float, npt.NDArray]:
    assert diag.ndim == 1

    if trial is None:
        trial = np.random.normal(size=diag.shape).astype(diag.dtype)
    trial = _normalize(trial)

    result = [trial]

    for i_step in range(max_steps):
        if np.linalg.norm(result[0]) < convergence_threshold:
            if verbose:
                print(f"(Converged at step {i_step})")
            break

        (theta, u), result = _davidson_step(
            matmul,
            diag,
            *result,
            thr=convergence_threshold,
        )

    return theta, u
