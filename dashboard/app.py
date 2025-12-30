from nicegui import ui
import pandas as pd
import plotly.express as px
import numpy as np
import os
import sys

# -------------------------------------------------------------------
# Veri yükleme
# -------------------------------------------------------------------
DATA_PATH = "../data/simulated/line_data.csv"
df = pd.read_csv(DATA_PATH)

# Son 200 adım gösterilecek (Dashboard'da çok daha net görünüyor)
tail_df = df.tail(200)


# -------------------------------------------------------------------
# Plot fonksiyonları
# -------------------------------------------------------------------
def plot_lead_time():
    fig = px.line(tail_df, x="time", y="lead_time",
                  title="Lead Time (Son 200 Adım)",
                  markers=True)
    return fig


def plot_queue():
    fig = px.line(tail_df, x="time", y="queue_length",
                  title="Queue Length (Son 200 Adım)",
                  markers=True)
    return fig


def plot_energy():
    fig = px.area(tail_df, x="time", y="energy_consumption",
                  title="Enerji Tüketimi")
    return fig


def plot_defects():
    fig = px.bar(tail_df, x="time", y="defects",
                 title="Hatalı Ürün Sayısı")
    return fig


# -------------------------------------------------------------------
# RL Agent Mock (Gerçek PPO modeliyle bağlanabilir)
# -------------------------------------------------------------------
def rl_action_suggestion(state_vector):
    """
    Şimdilik rastgele aksiyon üretiyoruz.
    Daha sonra gerçek PPO modelini load edip bağlayabilirsin:

    from stable_baselines3 import PPO
    model = PPO.load("../models/ppo_model.zip")
    action, _ = model.predict(obs)
    """
    actions = ["Hız Azalt", "Sabit", "Hız Artır"]
    return np.random.choice(actions)


# -------------------------------------------------------------------
# UI – Dashboard Tasarımı
# -------------------------------------------------------------------
ui.label("Üretim Hattı | RL + Zaman Serisi Kontrol Paneli") \
    .classes("text-3xl font-bold text-center mt-4")

ui.separator()

with ui.row().classes("w-full justify-center"):

    # ----------- Lead Time ------------
    with ui.card().classes("w-1/3 p-4"):
        ui.label("Lead Time").classes("text-xl font-bold")
        ui.plotly(plot_lead_time)

    # ----------- Queue Length ------------
    with ui.card().classes("w-1/3 p-4"):
        ui.label("Queue Length").classes("text-xl font-bold")
        ui.plotly(plot_queue)

    # ----------- Energy Consumption ------------
    with ui.card().classes("w-1/3 p-4"):
        ui.label("Enerji Tüketimi").classes("text-xl font-bold")
        ui.plotly(plot_energy)

ui.separator()

with ui.row().classes("w-full justify-center"):

    # ----------- Machine Status Cards ------------
    with ui.card().classes("w-1/4 p-4"):
        ui.label("Makine A Durum").classes("text-lg font-semibold")
        statusA = int(df["machine_A_status"].iloc[-1])
        ui.label("Çalışıyor" if statusA == 1 else "Arızalı") \
            .classes("text-xl text-green-600" if statusA == 1 else "text-xl text-red-600")

    with ui.card().classes("w-1/4 p-4"):
        ui.label("Makine B Durum").classes("text-lg font-semibold")
        statusB = int(df["machine_B_status"].iloc[-1])
        ui.label("Çalışıyor" if statusB == 1 else "Arızalı") \
            .classes("text-xl text-green-600" if statusB == 1 else "text-xl text-red-600")

    with ui.card().classes("w-1/4 p-4"):
        ui.label("Makine C Durum").classes("text-lg font-semibold")
        statusC = int(df["machine_C_status"].iloc[-1])
        ui.label("Çalışıyor" if statusC == 1 else "Arızalı") \
            .classes("text-xl text-green-600" if statusC == 1 else "text-xl text-red-600")

ui.separator()

# -------------------------------------------------------------------
# RL Aksiyon Öneri Bölümü
# -------------------------------------------------------------------
ui.label("RL Ajanı Aksiyon Önerisi").classes("text-2xl font-bold text-center mt-6")

with ui.card().classes("w-1/2 mx-auto p-6"):

    ui.label("Mevcut Durumu RL'e Gönder") \
        .classes("text-lg font-semibold")

    # State vector hazırlama
    last_state = df[["queue_length", "operator_load", "machine_status", "lead_time"]].iloc[-1]

    ui.label(f"State → {last_state.values}").classes("mt-2")

    if ui.button("Aksiyon Tahmin Et (PPO ile bağlanabilir)").classes("mt-4"):

        suggestion = rl_action_suggestion(last_state.values)

        ui.notify(f"RL Ajanı Önerisi: {suggestion}",
                  color="primary",
                  position="top")

ui.separator()

# -------------------------------------------------------------------
# Defect Chart
# -------------------------------------------------------------------
with ui.row().classes("w-full justify-center mb-10"):
    with ui.card().classes("w-1/2 p-4"):
        ui.label("Hatalı Ürün Trend").classes("text-xl font-bold")
        ui.plotly(plot_defects)

# -------------------------------------------------------------------
# Uygulamayı başlat
# -------------------------------------------------------------------
ui.run(port=8000, title="RL ve Zaman Serisi Dashboard")

