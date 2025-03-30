import logging 
import numpy as np
import pandas as pd
import ta
import gym
from gym import spaces
from datetime import datetime
import torch
import os
import optuna
import time

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from sklearn.preprocessing import MinMaxScaler

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for h in logger.handlers[:]:
    logger.removeHandler(h)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/run.log", mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(filepath: str):
    data = pd.read_csv(filepath)
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y.%m.%d %H:%M')
    data.set_index('Datetime', inplace=True)
    data.drop(['Date','Time'], axis=1, inplace=True)
    data.sort_index(inplace=True)
    return data

def add_features(df: pd.DataFrame):
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    df['Log_Return'] = np.log(df['Close']/df['Close'].shift(1))
    df.dropna(inplace=True)
    return df

class AggressiveTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, history_length=10, transaction_cost=0.001, spread=0.0002):
        super(AggressiveTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.entry_price = None
        self.current_step = 0
        self.max_step = len(df)-1
        self.transaction_cost = transaction_cost
        self.spread = spread

        self.features = ['Close','SMA_20','RSI_14','MACD','MACD_signal','SMA_200','ATR','Log_Return']
        data_feats = self.df[self.features].fillna(method='ffill').fillna(method='bfill')
        self.scaler = MinMaxScaler()
        self.scaler.fit(data_feats.values)

        self.history_length = history_length
        self.nb_features = len(self.features)
        obs_dim = (self.history_length * self.nb_features) + 1

        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE

        self.last_value = self.initial_balance
        self.price_history = []
        self.episode_returns = []
        self.steps_without_position = 0
        self.same_action_count = 0
        self.last_action = None
        self.max_balance = initial_balance

        self.latent_profit_threshold = 10.0  
        self.latent_loss_threshold = -10.0

        self.last_closed_position_type = None # Pour encourager alternance long/short
        self.successive_wins = 0 # Compter les enchainements de trades gagnants

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _get_current_features(self):
        row = self.df.iloc[self.current_step]
        feat_values = row[self.features].values.reshape(1,-1)
        scaled = self.scaler.transform(feat_values).flatten()
        return scaled

    def _get_obs(self):
        while len(self.price_history) < self.history_length:
            self.price_history.insert(0, self._get_current_features())

        hist = self.price_history[-self.history_length:]
        obs = np.concatenate(hist)
        pos_value = 0
        if self.position == 'LONG':
            pos_value = 1
        elif self.position == 'SHORT':
            pos_value = 2
        obs = np.append(obs, pos_value)
        return obs.astype(np.float32)

    def _get_portfolio_value(self):
        price = self.df['Close'].iloc[self.current_step]
        val = self.balance
        if self.position == 'LONG':
            val += (price - self.entry_price)
        elif self.position == 'SHORT':
            val += (self.entry_price - price)
        return val

    def step(self, action):
        self.current_step += 1
        if self.current_step > self.max_step:
            self.current_step = self.max_step

        self.price_history.append(self._get_current_features())
        done = (self.current_step >= self.max_step)
        price = self.df['Close'].iloc[self.current_step]

        reward = 0.0
        cost = 0.0
        realized_pnl = 0.0

        # Encourager haute fréquence : petit bonus si action != HOLD
        if action in [1,2,3]:
            reward += 0.1

        # Vérification d'action répétée
        if action == self.last_action:
            self.same_action_count += 1
        else:
            # Bonus si changement d'orientation (BUY->SELL ou SELL->BUY)
            if self.last_action is not None and action in [1,2] and self.last_action in [1,2] and action!=self.last_action:
                reward += 0.5
            self.same_action_count = 0
        self.last_action = action

        if action == 1: # BUY
            if self.position is None:
                trade_price = price * (1 + self.spread)
                cost = trade_price * self.transaction_cost
                self.position = 'LONG'
                self.entry_price = trade_price
                self.balance -= cost
        elif action == 2: # SELL
            if self.position is None:
                trade_price = price * (1 - self.spread)
                cost = trade_price * self.transaction_cost
                self.position = 'SHORT'
                self.entry_price = trade_price
                self.balance -= cost
        elif action == 3: # CLOSE
            if self.position == 'LONG':
                exit_price = price * (1 - self.spread)
                pnl = exit_price - self.entry_price
                cost = exit_price * self.transaction_cost
                self.balance += pnl - cost
                realized_pnl = pnl - cost
                if realized_pnl > 0:
                    self.successive_wins += 1
                else:
                    self.successive_wins = 0
                self.last_closed_position_type = 'LONG'
            elif self.position == 'SHORT':
                exit_price = price * (1 + self.spread)
                pnl = self.entry_price - exit_price
                cost = exit_price * self.transaction_cost
                self.balance += pnl - cost
                realized_pnl = pnl - cost
                if realized_pnl > 0:
                    self.successive_wins += 1
                else:
                    self.successive_wins = 0
                self.last_closed_position_type = 'SHORT'
            self.position = None
            self.entry_price = None

        current_val = self._get_portfolio_value()
        delta_val = current_val - self.last_value
        self.last_value = current_val

        # Récompense sur variation portefeuille
        reward += delta_val

        # Pénalité inaction prolongée sans position
        if self.position is None and action == 0:
            self.steps_without_position += 1
            reward -= 0.001 * self.steps_without_position
        else:
            self.steps_without_position = 0

        # Latent PNL
        if self.position is not None:
            latent_pnl = (price - self.entry_price) if self.position == 'LONG' else (self.entry_price - price)
            if latent_pnl > self.latent_profit_threshold:
                reward += 0.01 * (latent_pnl - self.latent_profit_threshold)
            if latent_pnl < self.latent_loss_threshold:
                reward += 0.01 * (latent_pnl - self.latent_loss_threshold)

        # Récompense sur clôture profitable
        if action == 3 and realized_pnl != 0:
            if realized_pnl > 0:
                # Récompense supplémentaire en fonction du nombre de gains successifs
                reward += realized_pnl * (2.0 + 0.1*self.successive_wins)
            else:
                reward += realized_pnl * 0.5

        # Pénalité action répétée
        if self.same_action_count > 5:
            reward -= 0.005 * self.same_action_count

        if self.balance > self.max_balance:
            self.max_balance = self.balance

        self.episode_returns.append(reward)

        if done:
            returns_arr = np.array(self.episode_returns)
            mean_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr) + 1e-8
            sharpe = mean_ret / std_ret
            reward += sharpe

            final_portfolio_val = self._get_portfolio_value()
            total_gain = final_portfolio_val - self.initial_balance
            reward += total_gain * 0.05

            drawdown = (self.max_balance - final_portfolio_val)
            reward -= drawdown * 0.02

        obs = self._get_obs()
        return obs, reward, done, {}

    def reset(self, **kwargs):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = None
        self.last_value = self.balance
        self.price_history = []
        self.episode_returns = []
        self.steps_without_position = 0
        self.same_action_count = 0
        self.last_action = None
        self.max_balance = self.initial_balance
        self.successive_wins = 0
        self.last_closed_position_type = None
        self.price_history.append(self._get_current_features())
        return self._get_obs()

    def render(self, mode='human'):
        pass

def evaluate_model(model, df, initial_balance=10000):
    env = AggressiveTradingEnv(df, initial_balance=initial_balance)
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward

class TrialLoggingCallback:
    def __init__(self):
        pass

    def __call__(self, study, trial):
        logging.info(f"Trial {trial.number} terminé. Value: {trial.value}, Params: {trial.params}, UserAttrs: {trial.user_attrs}")

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [2048, 4096, 8192])
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 512, step=64)
    ent_coef = trial.suggest_float('ent_coef', 1e-5, 1e-1, log=True)

    logging.info(f"Trial {trial.number} started with params: lr={lr}, n_steps={n_steps}, n_lstm_layers={n_lstm_layers}, lstm_hidden_size={lstm_hidden_size}, ent_coef={ent_coef}")

    policy_kwargs = dict(
        n_lstm_layers=n_lstm_layers,
        lstm_hidden_size=lstm_hidden_size
    )

    model = RecurrentPPO(
        MlpLstmPolicy,
        train_env,
        verbose=0,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=1024,
        policy_kwargs=policy_kwargs,
        device='cuda',
        ent_coef=ent_coef,
        seed=SEED
    )

    start_time = time.time()
    model.learn(total_timesteps=200000)
    train_time = time.time() - start_time

    total_reward = evaluate_model(model, val_df)
    logging.info(f"Trial {trial.number}: total_reward={total_reward}, LR={lr}, Steps={n_steps}, Layers={n_lstm_layers}, HSize={lstm_hidden_size}, ent_coef={ent_coef}, Train_time={train_time:.2f}s")

    trial.set_user_attr("train_time_s", train_time)
    trial.set_user_attr("final_reward_val", total_reward)

    return -total_reward

if __name__ == "__main__":
    DATA_PATH = 'data/raw/xauusd.csv'
    df = load_data(DATA_PATH)
    df = df.loc['2022-01-01':'2024-09-19']
    df = add_features(df)

    if len(df) < 2000:
        logging.error("Pas assez de données.")
        exit(1)

    # On pourrait mélanger les données, par ex:
    # Mais on garde la même logique
    train_df = df.loc[:'2023-12-31']
    val_df = df.loc['2024-01-01':'2024-06-30']
    test_df = df.loc['2024-07-01':]

    train_env = AggressiveTradingEnv(train_df)
    train_env = DummyVecEnv([lambda: train_env])

    n_trials = 100
    import optuna
    study = optuna.create_study(direction="minimize")

    callback = TrialLoggingCallback()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    best_params = study.best_params
    logging.info(f"Meilleurs hyperparamètres trouvés: {best_params}")

    final_policy_kwargs = dict(
        n_lstm_layers=best_params['n_lstm_layers'],
        lstm_hidden_size=best_params['lstm_hidden_size']
    )

    val_env = AggressiveTradingEnv(val_df)
    val_env = DummyVecEnv([lambda: val_env])

    eval_freq = 50000
    patience_evaluations = 5
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience_evaluations, verbose=1)
    eval_cb = EvalCallback(val_env, eval_freq=eval_freq, n_eval_episodes=3, callback_after_eval=stop_cb, verbose=1)
    checkpoint_cb = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='rl_model')

    final_timesteps = 2_000_000
    model = RecurrentPPO(
        MlpLstmPolicy,
        train_env,
        verbose=1,
        learning_rate=best_params['lr'],
        n_steps=best_params['n_steps'],
        batch_size=1024,
        policy_kwargs=final_policy_kwargs,
        device='cuda',
        ent_coef=best_params['ent_coef'],
        seed=SEED
    )
    model.learn(total_timesteps=final_timesteps, callback=[eval_cb, checkpoint_cb])

    final_reward = evaluate_model(model, test_df)
    logging.info(f"Reward final sur test set: {final_reward}")

    best_trial = study.best_trial
    best_trial_data = {
        'number': [best_trial.number],
        'value': [best_trial.value],
        'params': [str(best_trial.params)],
        'user_attrs': [str(best_trial.user_attrs)]
    }
    best_df = pd.DataFrame(best_trial_data)
    os.makedirs("logs", exist_ok=True)
    best_df.to_csv("logs/best_trial_only.csv", index=False)
    logging.info("Meilleur trial sauvegardé dans logs/best_trial_only.csv")

    os.makedirs("models", exist_ok=True)
    model.save("models/best_model.zip")
    logging.info("Meilleur modèle sauvegardé dans models/best_model.zip")