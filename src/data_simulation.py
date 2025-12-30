"""
============================================================
 ENDÜSTRİYEL ÜRETİM HATTI – GELİŞMİŞ DİJİTAL İKİZ SİMÜLASYONU
============================================================

Bu modül, üretim hattı için yüksek gerçekçilik seviyesinde bir
dijital ikiz (digital twin) zaman serisi üretir.

İçerdiği tüm özellikler:
- 3 Makine (A, B, C) → farklı hız, arıza olasılığı, bakım döngüsü
- Vardiya sistemi → gündüz, akşam, gece
- Saatlik günlük pattern (sinüzoidal talep)
- Haftalık pattern
- Demand spike (patlama)
- Batch arrival (toplu iş)
- Operatör skill (öğrenme eğrisi) & fatigue (yorgunluk)
- Arıza – downtime (Poisson)
- Periyodik bakım (interval tabanlı)
- Bakım gecikirse arıza risk artışı
- Normal + öncelikli kuyruk (WIP)
- Defect → rework döngüsü (yeniden kuyruğa ekleme)
- Enerji tüketimi
- RL ile uyumlu kolonlar: queue_length, lead_time, machine_status

Kullanım:
---------
from src.data_simulation import simulate_production_line_advanced

df = simulate_production_line_advanced(T=2000)
df.head()

============================================================
"""

import numpy as np
import pandas as pd


# =====================================================================
#  ANA GELİŞMİŞ ÜRETİM HATTI SİMÜLASYON FONKSİYONU
# =====================================================================
def simulate_production_line_advanced(
    T: int = 2000,
    dt: float = 1.0,
    base_arrival: float = 2.0,
    base_service_A: float = 2.5,
    base_service_B: float = 2.0,
    base_service_C: float = 1.5,
    breakdown_prob_A: float = 0.015,
    breakdown_prob_B: float = 0.02,
    breakdown_prob_C: float = 0.03,
    avg_downtime_A: int = 20,
    avg_downtime_B: int = 30,
    avg_downtime_C: int = 40,
    maintenance_interval_A: int = 400,
    maintenance_interval_B: int = 500,
    maintenance_interval_C: int = 600,
    operator_fatigue_rate: float = 0.004,
    defect_base: float = 0.03,
    energy_idle: float = 1.0,
    energy_per_speed: float = 0.8,
    noise_level: float = 0.15,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Gelişmiş üretim hattı dijital ikiz simülasyonu.
    """

    if seed is not None:
        np.random.seed(seed)

    steps = np.arange(T)
    times = steps * dt

    # Operatör parametreleri
    operator_skill = 0.4
    operator_fatigue = 0.2
    operator_load = 0.7

    # Kuyruklar
    normal_queue = 0.0
    priority_queue = 0.0

    # Makine parametre setleri
    machines = ["A", "B", "C"]
    status = {m: 1 for m in machines}
    downtime_timer = {m: 0 for m in machines}
    maintenance_timer = {m: 0 for m in machines}

    base_service = {"A": base_service_A, "B": base_service_B, "C": base_service_C}
    breakdown_prob = {"A": breakdown_prob_A, "B": breakdown_prob_B, "C": breakdown_prob_C}
    avg_downtime = {"A": avg_downtime_A, "B": avg_downtime_B, "C": avg_downtime_C}
    maintenance_interval = {
        "A": maintenance_interval_A,
        "B": maintenance_interval_B,
        "C": maintenance_interval_C,
    }

    # Kayıt yapısı
    cols = [
        "time","step","hour","shift_id",
        "normal_queue","priority_queue","wip_total",
        "completed_jobs","defect_rate","defects",
        "energy_consumption",
        "operator_load","operator_skill","operator_fatigue",
        "machine_A_status","machine_B_status","machine_C_status",
        "machine_A_speed","machine_B_speed","machine_C_speed",
        "maintenance_A","maintenance_B","maintenance_C",
        "demand_spike_flag",
    ]
    records = {c: [] for c in cols}

    # ============================================================
    #  ANA ZAMAN DÖNGÜSÜ
    # ============================================================
    for idx, t in enumerate(steps):

        current_time = times[idx]
        hour = (t % 144) // 6

        # Vardiya
        if 6 <= hour < 14:
            shift_id = 1; shift_factor = 1.20
        elif 14 <= hour < 22:
            shift_id = 2; shift_factor = 1.00
        else:
            shift_id = 3; shift_factor = 0.85

        # Arrival
        daily_pattern = 1 + 0.3*np.sin(2*np.pi*(t/144))
        weekly_pattern = 1 + 0.2*np.sin(2*np.pi*(t/1008))

        if np.random.rand() < 0.02:
            spike_flag = 1
            spike_factor = 2.5
        else:
            spike_flag = 0
            spike_factor = 1.0

        batch = np.random.choice([0,5,10], p=[0.85,0.1,0.05])

        lam = base_arrival * daily_pattern * weekly_pattern * shift_factor * spike_factor * dt
        arrivals = np.random.poisson(lam) + batch

        pr_ratio = 0.15 + 0.1*np.sin(2*np.pi*(t/288))
        pr_in = int(arrivals * pr_ratio)
        nr_in = arrivals - pr_in

        normal_queue += nr_in
        priority_queue += pr_in

        # Operatör dinamiği
        time_in_shift = t % 48
        skill_progress = 1 / (1 + np.exp(-0.1*(time_in_shift - 24)))
        operator_skill = 0.3 + 0.7*skill_progress

        operator_fatigue += operator_fatigue_rate * dt
        operator_fatigue = float(np.clip(operator_fatigue, 0.1, 1.0))

        if time_in_shift == 0 and t > 0:
            operator_fatigue = 0.3 + np.random.rand()*0.1

        operator_load = float(
            np.clip(0.5 + 0.4*operator_fatigue - 0.2*(operator_skill - 0.5), 0.1, 1.0)
        )

        # Makine hızları
        machine_speed = {}
        maintenance_flag = {"A":0,"B":0,"C":0}

        for m in machines:

            if downtime_timer[m] > 0:
                status[m] = 0
                downtime_timer[m] -= 1

            elif maintenance_timer[m] > 0:
                status[m] = 0
                maintenance_timer[m] -= 1
                maintenance_flag[m] = 1

            else:
                if t > 0 and t % maintenance_interval[m] == 0:
                    if np.random.rand() < 0.7:
                        maintenance_timer[m] = np.random.randint(5,20)
                        status[m] = 0
                        maintenance_flag[m] = 1
                    else:
                        if np.random.rand() < breakdown_prob[m]*2:
                            status[m] = 0
                            downtime_timer[m] = np.random.poisson(avg_downtime[m])
                        else:
                            status[m] = 1
                else:
                    if np.random.rand() < breakdown_prob[m]:
                        status[m] = 0
                        downtime_timer[m] = np.random.poisson(avg_downtime[m])
                    else:
                        status[m] = 1

            # Hız
            if status[m] == 1:
                speed = (
                    base_service[m]
                    * shift_factor
                    * (0.8 + 0.4*operator_skill)
                    * (1.1 - 0.3*operator_fatigue)
                )
                speed = max(speed + np.random.randn()*noise_level, 0)
            else:
                speed = 0.0

            machine_speed[m] = speed

        # İşleme
        total_completed = 0
        total_defects = 0

        for m in machines:

            if machine_speed[m] <= 0:
                continue

            cap = int(max(round(machine_speed[m]*dt), 0))

            from_pr = min(priority_queue, cap)
            priority_queue -= from_pr

            remaining = cap - from_pr
            from_nr = min(normal_queue, remaining)
            normal_queue -= from_nr

            processed = from_pr + from_nr
            if processed <= 0:
                continue

            avg_status = np.mean([status[x] for x in machines])

            defect_rate = defect_base
            defect_rate += 0.1*operator_fatigue
            defect_rate += 0.05*(1-operator_skill)
            defect_rate += 0.05*(1-avg_status)
            defect_rate = float(np.clip(defect_rate, 0, 0.4))

            defects = np.random.binomial(processed, defect_rate)

            normal_queue += defects  # rework

            total_completed += (processed - defects)
            total_defects += defects

        wip_total = normal_queue + priority_queue

        # Enerji
        energy = 0
        for m in machines:
            if status[m] == 0:
                energy += energy_idle*0.3*dt
            else:
                energy += (energy_idle + energy_per_speed*machine_speed[m]) * dt

        # Kayıt
        rec = records
        rec["time"].append(current_time)
        rec["step"].append(int(t))
        rec["hour"].append(int(hour))
        rec["shift_id"].append(int(shift_id))
        rec["normal_queue"].append(float(normal_queue))
        rec["priority_queue"].append(float(priority_queue))
        rec["wip_total"].append(float(wip_total))
        rec["completed_jobs"].append(int(total_completed))
        rec["defect_rate"].append(float(defect_rate if total_completed > 0 else 0))
        rec["defects"].append(int(total_defects))
        rec["energy_consumption"].append(float(energy))
        rec["operator_load"].append(float(operator_load))
        rec["operator_skill"].append(float(operator_skill))
        rec["operator_fatigue"].append(float(operator_fatigue))
        rec["machine_A_status"].append(int(status["A"]))
        rec["machine_B_status"].append(int(status["B"]))
        rec["machine_C_status"].append(int(status["C"]))
        rec["machine_A_speed"].append(float(machine_speed["A"]))
        rec["machine_B_speed"].append(float(machine_speed["B"]))
        rec["machine_C_speed"].append(float(machine_speed["C"]))
        rec["maintenance_A"].append(int(maintenance_flag["A"]))
        rec["maintenance_B"].append(int(maintenance_flag["B"]))
        rec["maintenance_C"].append(int(maintenance_flag["C"]))
        rec["demand_spike_flag"].append(int(spike_flag))

    # ============================================================
    #  ÇIKTI DATAFRAME
    # ============================================================
    df = pd.DataFrame(records)

    df["queue_length"] = df["wip_total"]
    safe_completed = df["completed_jobs"].replace(0, np.nan)
    df["lead_time"] = (df["wip_total"] / safe_completed).fillna(method="ffill").fillna(
        df["wip_total"].mean()
    )

    df["machine_status"] = (
        df["machine_A_status"]
        + df["machine_B_status"]
        + df["machine_C_status"]
    ) / 3

    return df


# =====================================================================
#  GERİYE DÖNÜK UYUMLULUK
# =====================================================================
def simulate_production_line_realistic(T: int = 2000, **kwargs) -> pd.DataFrame:
    return simulate_production_line_advanced(T=T, **kwargs)
