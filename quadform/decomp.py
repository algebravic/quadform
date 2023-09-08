"""
Find a decomposition $A A^T = G$ of an integral positive definite
Gram matrix using nonconvex optimization over the orthogonal group.
"""
from typing import Iterable
from time import time
from functools import partial
import torch
import geoopt
from landing import LandingSGD
import numpy as np
import matplotlib.pyplot as plt
import scipy

def cos_loss(params: np.ndarray, mat: np.ndarray, power: int = 1):
    return torch.sum(params * (1 - torch.cos( 2 * torch.pi * mat)))

def sin_loss(params: np.ndarray, mat: np.ndarray, power: int = 4):
    
    return torch.sum(params * torch.pow(torch.abs(torch.sin(torch.pi * mat)), power))

def abs_loss(params: np.ndarray, mat: np.ndarray, power: int = 1):
    return torch.sum(params
                     * torch.pow(
                         torch.abs(mat - torch.floor(mat)), power))


LOSS_FN = {'cos': cos_loss, 'sin': sin_loss, 'abs': abs_loss}

def optimize(mat_, init, loss_fn = cos_loss, learning_rate = 0.3):
    """
p    Inputs:
       mat_: a numpy array of shape (n, m)
       init_weights: initial value for the orthogonal
          matrix (don't need to be orthogonal)
       Doesn't even have to be feasible.
    """
    # torch doesn't like float64 (maybe not)
    mat = torch.from_numpy(mat_.astype(np.float64).copy())
    init_weights = torch.from_numpy(init.astype(np.float64).copy())

    param = geoopt.ManifoldParameter(
        init_weights, manifold=geoopt.Stiefel(canonical=False))

    with torch.no_grad():
        param.proj_() # Projects onto the manifold

    optimizer = LandingSGD((param,), lr = learning_rate)

    start = time()
    while True: # Calling program determines when to stop
        optimizer.zero_grad()
        loss = loss_fn(param @ mat)
        loss.backward()
        yield (time() - start, loss.item(), param.data.clone())
        optimizer.step()

def show_errors(mat: np.ndarray):

    sorted_errors = np.abs(mat.reshape((-1,))).sort()
    plt.plot(sorted_errors)
    plt.show()

def projection(mat: np.ndarray) -> np.ndarray:
    """
    Closest orthogonal matrix in the Frobenius norm
    """

    if not (isinstance(mat, np.ndarray)
            and len(mat.shape) == 2
            and mat.shape[0] == mat.shape[1]):
        raise ValueError("Input must be a square matrix")

    umat, _, vmat = np.linalg.svd(mat)

    return umat @ vmat

def plot_progress(time_list, loss_list, dist_list,
                  method_name='quad'):

    fig, axes = plt.subplots(2, 1)
    axes[0].semilogy(time_list, dist_list, label=method_name)
    axes[0].set_xlabel("time (s.)")
    axes[0].set_ylabel("Orthogonality Error")
    axes[1].semilogy(time_list, loss_list, label=method_name)
    axes[1].set_xlabel("time (s.)")
    axes[1].set_ylabel("Error Error")
    plt.legend()
    
def recover_gram(grm : np.ndarray,
                 loss_name: str = 'cos',
                 power: int = 4,
                 extra: int = 5,
                 trace: int = 0,
                 verbose: int = 0,
                 hist: bool = False,
                 learning_rate: float = 0.3,
                 precision: float = 1.0e-3,
                 small: float = 1.0e-10,
                 discount: float = 0.95,
                 itmin: int = 500,
                 tries: int = 10) -> Iterable[np.ndarray]:
    """
    Recover a basis for the lattice generating a gram matrix
    """
    dim = grm.shape[0]
    umat, dval, vmat = np.linalg.svd(grm, full_matrices=True)
    gsqrt = umat @ np.diag(np.sqrt(dval)) @ vmat
    mdim = dim + extra
    res = np.linalg.qr(gsqrt.astype(np.float64), mode='complete')
    bigR = np.concatenate([res.R.T, np.zeros((dim, extra), dtype=np.float64)], axis = 1)
    # Use all 1's for now.  Maybe something cleverer later
    lparams = torch.ones(bigR.T.shape, dtype = torch.float64)
    loss_big = LOSS_FN.get(loss_name, LOSS_FN['cos'])
    loss_fn = partial(loss_big, lparams, power = power)

    for try_num in range(tries):

        init = np.random.randn(mdim, mdim)

        compute = optimize(bigR.T,
                           init,
                           learning_rate = learning_rate,
                           loss_fn = loss_fn)

        iteration = 0
        total_loss = 1.0
        denom = 1.0
        best_loss = None
        best_mat = None
        if verbose > 1:
            time_list = []
            loss_list = []
            distance_list = []
        while True:
            delta, loss, omat_ = next(compute)
            if best_loss is None or best_loss > loss:
                best_loss = loss
                best_mat = omat_.cpu().numpy().copy()
            iteration += 1
            if verbose > 1:
                time_list.append(delta)
                loss_list.append(loss)
                mdiff = omat_ @ torch.t(omat_) - torch.eye(mdim)
                distance_list.append(torch.norm(mdiff))
            scaled_loss = loss / (mdim * dim)
            total_loss = discount * total_loss + scaled_loss
            denom = discount * denom + 1.0
            if trace > 0 and (iteration % trace == 0):
                print(f"iteration = {iteration}, loss = {scaled_loss}")

            if ((scaled_loss <= precision)
                or
                (iteration > itmin
                 and abs(total_loss / denom - scaled_loss) < small)):
                break

        omat = projection(best_mat)
        
        target = (omat @ bigR.T).round()
        yield target
        
        guess = target.T @ target
        deltas = np.abs(guess - grm).flatten()
        print(f"try = {try_num}, {iteration} iterates, "
              + f"time = {delta}, dist = {deltas.mean()}")
        if verbose > 1:
            plot_progress(time_list, loss_list,
                          method_name = loss_fn_name)
        if verbose > 0:
            print(f"errors = {scipy.stats.describe(deltas)}")
