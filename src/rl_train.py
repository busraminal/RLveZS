from stable_baselines3 import PPO
from src.env_rl import ProductionLineEnv

# ============================================================
# PPO AJANI EĞİTİMİ
# ============================================================

def train_ppo(env):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)   # eğitim süresi
    model.save("ppo_model")               # model kaydı
    return model
