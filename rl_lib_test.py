from simple_model_env import SimpleCropModelEnv
from simple import BentonWACSVWeatherSchedule, WeatherForecastSTDs, PotatoRussetUSACropParametersSpec

import datetime

import gym
import os
import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import ray.rllib.agents.sac as sac
import ray.rllib.agents.ppo as ppo
from ray.tune.schedulers import ASHAScheduler
import numpy as np


def register():
    register_env("SimpleModelEnv", lambda cfg: SimpleCropModelEnv(**cfg))
    # ModelCatalog.register_custom_model("atari", AtariNetwork)


def train_on_simple():
    ray.init(ignore_reinit_error=True,
             num_cpus=2)

    # config = sac.DEFAULT_CONFIG.copy()
    config = ppo.DEFAULT_CONFIG.copy()

    register()
    env_name = "SimpleModelEnv"

    config['env'] = env_name
    # config['model'] = dict(custom_model="atari", custom_model_config={'policy_dim': 6})
    config['framework'] = 'torch'

    config["num_workers"] = 1  # from ETN defaults
    # config['lr'] = 0.00025  # from ETN defaults
    # config['lambda'] = 0.95  # from ETN defaults
    # config["timesteps_per_iteration"] = 64
    # config['vf_loss_coeff'] = ray.tune.grid_search([0.5, 0.8, 1.0])  # from ETN defaults
    # config['clip_param'] = 0.2  # from ETN defaults


    # PPO config

    csv_path = '/Users/ashcrcc1/Downloads/Daily Data  AgWeatherNet at Washington State University.csv'
    env_cfg = dict(
        sowing_date=datetime.datetime(day=1, month=5, year=2021),
        num_growing_days=120,
        weather_schedule=BentonWACSVWeatherSchedule(csv_path),
        weather_forecast_stds=WeatherForecastSTDs(),
        latitude=43.9271,
        elevation=1464,
        crop_parameters=PotatoRussetUSACropParametersSpec(),
    )

    config["env_config"] = env_cfg

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=8000,
        grace_period=750,  # min time-steps to run before considering termination
        reduction_factor=4,
        brackets=1
    )

    # ray.tune.run("SAC",
    #              local_dir='./ray_results/',
    #              stop={"timesteps_total": 1e6, "episode_reward_mean": 95},
    #              checkpoint_freq=10000,
    #              checkpoint_at_end=True,
    #              config=config)

    ray.tune.run("PPO",
                 local_dir='./ray_results/',
                 stop={"timesteps_total": 1e6, "episode_reward_mean": 50000},
                 checkpoint_freq=10000,
                 checkpoint_at_end=True,
                 config=config)


if __name__ == "__main__":
    train_on_simple()