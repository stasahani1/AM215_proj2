"""GLM model for infection hazard and recovery rate estimation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import cloglog
from dataclasses import dataclass
from dataset import InfectionDataset


@dataclass
class InfectionModel:
    """Container for fitted GLM model."""
    result: sm.GLM


def fit_cloglog_glm(dataset: InfectionDataset) -> InfectionModel:
    """
    Fit complementary log-log GLM for S->I transitions.

    The cloglog link is natural for hazard modeling:
    P(infection) = 1 - exp(-exp(X*beta))

    Parameters
    ----------
    dataset : InfectionDataset
        Training dataset with features X and labels y

    Returns
    -------
    InfectionModel
        Fitted model container
    """
    model = sm.GLM(dataset.y, dataset.X, family=Binomial(link=cloglog()))
    res = model.fit()
    return InfectionModel(result=res)


def predict_prob(model: InfectionModel, X: pd.DataFrame) -> np.ndarray:
    """
    Predict infection probabilities for given features.

    Parameters
    ----------
    model : InfectionModel
        Fitted infection model
    X : pd.DataFrame
        Feature matrix (should have same columns as training data)

    Returns
    -------
    np.ndarray
        Predicted infection probabilities
    """
    return model.result.predict(X)


def estimate_recovery_hazard(I: pd.DataFrame) -> float:
    """
    Estimate daily recovery probability mu via geometric MLE: p = 1 / mean_duration.
    (duration measured in consecutive days of I=1).

    Parameters
    ----------
    I : pd.DataFrame
        Infection state indicators

    Returns
    -------
    float
        Estimated daily recovery probability
    """
    durations = []
    for col in I.columns:
        arr = I[col].values.astype(int)
        t = 0
        while t < len(arr):
            while t < len(arr) and arr[t] == 0:
                t += 1
            if t >= len(arr):
                break
            start = t
            while t < len(arr) and arr[t] == 1:
                t += 1
            durations.append(t - start)

    if len(durations) == 0:
        return 0.2  # fallback

    mean_len = np.mean(durations)
    mu = 1.0 / mean_len  # geometric MLE

    return float(np.clip(mu, 1e-4, 0.99))
