import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ============================================================
# METRİKLER
# ============================================================

def compute_mape_rmse(true, pred):
    mape = mean_absolute_percentage_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mape, rmse


def success_rate(success_flags):
    return sum(success_flags) / len(success_flags)


def discounted_reward(rewards, gamma=0.99):
    total = 0
    for t, r in enumerate(rewards):
        total += (gamma ** t) * r
    return total
