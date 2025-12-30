import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ============================================================
# ZAMAN SERİSİ MODELLERİ
# Amaç: lead_time tahmini yapıp MAPE & RMSE hesaplamak
# ============================================================

def sarima_forecast(df):
    """
    lead_time değişkenini SARIMA ile tahmin eder.
    """

    # Model tanımı (örnek parametreler)
    model = SARIMAX(
        df["lead_time"],
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 12)
    )

    # Eğit
    result = model.fit(disp=False)

    # Test seti: veri setinin son yarısı
    start = len(df) // 2
    end = len(df) - 1

    preds = result.predict(start=start, end=end)
    true = df["lead_time"].iloc[start:end+1]

    # Metrikler
    mape = mean_absolute_percentage_error(true, preds)
    rmse = np.sqrt(mean_squared_error(true, preds))

    return preds, mape, rmse
