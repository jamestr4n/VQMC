"""Stage 1, Day 2: finite-difference second derivatives and local energy (1D SHO).

Natural units: x in units of x0 = sqrt(hbar/(m*omega)), E in units of hbar*omega.
Hamiltonian: H = -1/2 d^2/dx^2 + 1/2 x^2
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from math import factorial, pi, sqrt


#1. Finite-difference second derivatives 
def d2_3pt(f, x, h):
    """3 point central difference, truncation error O(h^2)."""
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / h**2


def d2_5pt(f, x, h):
    """5 point central difference, truncation error O(h^4)."""
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12.0 * h**2)


#2. Test functions
def psi0(x):
    return np.exp(-x**2 / 2)


def psi0_dd_exact(x):
    return (x**2 - 1) * np.exp(-x**2 / 2)


def psi_n(n):
    """Return the (unnormalised-is-fine) n-th SHO eigenfunction H_n(x) e^{-x^2/2}."""
    c = np.zeros(n + 1); c[n] = 1.0
    norm = 1.0 / sqrt(2**n * factorial(n) * sqrt(pi))
    return lambda x: norm * hermval(x, c) * np.exp(-x**2 / 2)


#3. Local energy
def local_energy(psi, x, h=1e-3, method=d2_5pt):
    """E_L(x) = (H psi)(x) / psi(x) = -1/2 psi''/psi + 1/2 x^2."""
    return -0.5 * method(psi, x, h) / psi(x) + 0.5 * x**2


if __name__ == "__main__":
    #Error vs h at a fixed point
    x_test = 0.5
    hs = np.logspace(-8, 0, 200)
    exact = psi0_dd_exact(x_test)
    err3 = np.abs(d2_3pt(psi0, x_test, hs) - exact)
    err5 = np.abs(d2_5pt(psi0, x_test, hs) - exact)

    i3, i5 = np.argmin(err3), np.argmin(err5)
    print(f"3-point: best h = {hs[i3]:.2e}, min error = {err3[i3]:.2e}")
    print(f"5-point: best h = {hs[i5]:.2e}, min error = {err5[i5]:.2e}")

    # fitted slopes in the truncation-dominated region
    m3 = (hs > 1e-2) & (hs < 1e-1)
    m5 = (hs > 3e-2) & (hs < 2e-1)
    print("slope 3-pt:", np.polyfit(np.log(hs[m3]), np.log(err3[m3]), 1)[0])
    print("slope 5-pt:", np.polyfit(np.log(hs[m5]), np.log(err5[m5]), 1)[0])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(hs, err3, label="3-point")
    ax[0].loglog(hs, err5, label="5-point")
    ax[0].loglog(hs, 0.1 * hs**2, "k--", lw=0.8, label=r"$\propto h^2$")
    ax[0].loglog(hs, 0.01 * hs**4, "k:", lw=0.8, label=r"$\propto h^4$")
    ax[0].loglog(hs, 1e-16 / hs**2, "r-.", lw=0.8, label=r"$\propto \epsilon/h^2$")
    ax[0].set_ylim(1e-14, 1)
    ax[0].set_xlabel("step size $h$"); ax[0].set_ylabel(r"$|\psi_0''_{FD}-\psi_0''|$")
    ax[0].set_title(f"FD error at x = {x_test}"); ax[0].legend(fontsize=8)

    # Local energy for n = 0..3
    x = np.linspace(-4, 4, 801)
    for n in range(4):
        EL = local_energy(psi_n(n), x)
        ax[1].plot(x, EL, label=f"n={n}")
        ok = np.abs(psi_n(n)(x)) > 1e-3   # skip points at/near nodes
        print(f"n={n}: mean E_L away from nodes = {EL[ok].mean():.8f}, "
              f"max dev = {np.abs(EL[ok]-(n+0.5)).max():.1e} (exact {n+0.5})")
    ax[1].set_ylim(-1, 5)
    ax[1].set_xlabel("x"); ax[1].set_ylabel(r"$E_L(x)$")
    ax[1].set_title("Local energy (5-pt, h = 1e-3)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fd_day2.png", dpi=150)