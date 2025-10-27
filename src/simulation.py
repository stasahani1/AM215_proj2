"""SIR simulation for scenario analysis."""
import numpy as np
from typing import Tuple


def simulate_sir(
    S0: np.ndarray,
    I0: np.ndarray,
    R0: np.ndarray,
    W: np.ndarray,
    beta: float,
    mu: float,
    market_path: np.ndarray,
    steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple stochastic daily SIR on a static network.

    Uses the hazard model:
      lambda_i,t = 1 - exp(-(beta * (W I_t)_i + market_t))

    I->R transitions occur with probability mu.
    R stays R (no waning) for simplicity here.

    Parameters
    ----------
    S0 : np.ndarray
        Initial susceptible state (n,)
    I0 : np.ndarray
        Initial infected state (n,)
    R0 : np.ndarray
        Initial recovered state (n,)
    W : np.ndarray
        Network weight matrix (n, n)
    beta : float
        Infection strength parameter
    mu : float
        Recovery probability per day
    market_path : np.ndarray
        Market stress values for each simulation step
    steps : int
        Number of simulation steps (default: 30)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        S_path, I_path, R_path each of shape (steps+1, n)
    """
    n = len(S0)
    S, I, R = S0.copy(), I0.copy(), R0.copy()
    S_path, I_path, R_path = [S.copy()], [I.copy()], [R.copy()]

    for t in range(steps):
        exposure = W @ I
        market_t = market_path[min(t, len(market_path)-1)]
        lam = 1.0 - np.exp(-(beta * exposure + market_t))

        # Transitions
        s2i = (np.random.rand(n) < lam) & (S == 1)
        i2r = (np.random.rand(n) < mu) & (I == 1)

        # Apply state changes
        S[s2i] = 0
        I[s2i] = 1
        I[i2r] = 0
        R[i2r] = 1

        S_path.append(S.copy())
        I_path.append(I.copy())
        R_path.append(R.copy())

    return np.array(S_path), np.array(I_path), np.array(R_path)


def run_scenario(
    S_snap: np.ndarray,
    I_snap: np.ndarray,
    R_snap: np.ndarray,
    W_snap: np.ndarray,
    beta: float,
    mu: float,
    market_scenario: str = "stressed",
    steps: int = 30,
    n_simulations: int = 100,
) -> dict:
    """
    Run multiple simulations under a given scenario and aggregate results.

    Parameters
    ----------
    S_snap, I_snap, R_snap : np.ndarray
        Initial states
    W_snap : np.ndarray
        Network weights
    beta : float
        Infection strength
    mu : float
        Recovery rate
    market_scenario : str
        "normal", "stressed", or "severe"
    steps : int
        Simulation horizon
    n_simulations : int
        Number of Monte Carlo runs

    Returns
    -------
    dict
        Summary statistics including peak infections, final recovered, etc.
    """
    # Define market stress paths
    scenarios = {
        "normal": np.zeros(steps),
        "stressed": np.r_[np.full(10, 2.0), np.zeros(steps - 10)],
        "severe": np.r_[np.full(15, 3.0), np.full(15, 1.0), np.zeros(max(0, steps - 30))]
    }
    market_path = scenarios.get(market_scenario, np.zeros(steps))

    peak_infections = []
    final_recovered = []

    for _ in range(n_simulations):
        S_path, I_path, R_path = simulate_sir(
            S_snap.copy(), I_snap.copy(), R_snap.copy(),
            W_snap, beta, mu, market_path, steps
        )
        peak_infections.append(I_path.sum(axis=1).max())
        final_recovered.append(R_path[-1].sum())

    return {
        "scenario": market_scenario,
        "peak_infected_mean": np.mean(peak_infections),
        "peak_infected_std": np.std(peak_infections),
        "peak_infected_95th": np.percentile(peak_infections, 95),
        "final_recovered_mean": np.mean(final_recovered),
        "initial_infected": I_snap.sum(),
        "initial_susceptible": S_snap.sum(),
        "steps": steps,
        "n_simulations": n_simulations
    }
