import copy

import numpy as np
import datetime
import gym

from simple_model_functions import calc_arid, calc_yield, calc_ref_evapotranspiration, calc_transpiration, \
    calc_f_temp, calc_f_co2, calc_f_heat, calc_f_water, calc_plant_available_water, calc_f_solar_water, calc_f_solar, \
    calc_biomass_rate, calc_cumulative_biomass, calc_rad_50p_senescence, delta_cumulative_temp


class SimpleCropModelEnv(gym.Env):
    def __init__(self, sowing_date,
                 num_growing_days,
                 weather_schedule,
                 weather_forecast_stds,
                 latitude,
                 elevation,
                 crop_parameters,
                 initial_biomass=0,  # Kg
                 initial_cumulative_temp=0,  # degrees C day
                 # initial_f_solar=0,  # unit-less; todo: paper says this is an init param, but where would it be used?
                 initial_available_water_content=None,  # not an initial parameter in SIMPLE, but reasonable to vary,
                 #  https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286 assumes initial value to
                 #  be equal to the available water capacity (AWC) in their crop model comparison
                 seed=None,
                 cumulative_biomass_std=1.0,
                 plant_available_water_std=1.0):

        # simulation parameters
        self.num_growing_days = num_growing_days

        # location parameters
        self.latitude = latitude
        self.elevation = elevation

        # crop parameters
        self.crop_temp_base = crop_parameters.temp_base
        self.crop_temp_opt = crop_parameters.temp_opt
        self.crop_RUE = crop_parameters.RUE
        self.crop_rad_50p_growth = crop_parameters.rad_50p_growth
        self.crop_rad_50p_senescence = crop_parameters.rad_50p_senescence
        self.crop_maturity_temp = crop_parameters.maturity_temp
        self.crop_rad_50p_max_heat = crop_parameters.rad_50p_max_heat
        self.crop_rad_50p_max_water = crop_parameters.rad_50p_max_water
        self.crop_heat_stress_thresh = crop_parameters.heat_stress_thresh
        self.crop_heat_stress_extreme = crop_parameters.heat_stress_extreme
        self.crop_drought_stress_sensitivity = crop_parameters.drought_stress_sensitivity
        self.crop_deep_drainage_coef = crop_parameters.deep_drainage_coef
        self.crop_water_holding_capacity = crop_parameters.water_holding_capacity
        self.crop_runoff_curve_number = crop_parameters.runoff_curve_number
        self.crop_root_zone_depth = crop_parameters.root_zone_depth
        self.crop_co2_sensitivity = crop_parameters.co2_sensitivity
        self.crop_harvest_index = crop_parameters.harvest_index

        # variables for reset
        self.init_cumulative_temp = initial_cumulative_temp
        self.init_crop_rad_50p_senescence = crop_parameters.rad_50p_senescence
        self.init_biomass = initial_biomass
        self.sowing_date = sowing_date

        # root zone available water (W_i) = root zone available water content (theta^{ad}_{a, i}) times the root
        #   zone depth (RZD), i.e. W_i =  theta^{ad}_{a, i} * RZD. See:
        #   https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286, which also justifies the initial
        #   condition shown here in the crop comparison section
        if initial_available_water_content is None:
            initial_available_water_content = self.crop_water_holding_capacity
        self.initial_available_water_content = initial_available_water_content  # unused, but saved for reference
        self.initial_plant_available_water = initial_available_water_content * self.crop_root_zone_depth

        # tracking variables
        self.cumulative_mean_temp = None  # TT variable in paper
        self.cumulative_biomass = None
        self.plant_available_water = None
        self.date = None
        self.day = None

        # Randomization parameters
        self.cumulative_biomass_sigma = cumulative_biomass_std
        self.plant_available_water_sigma = plant_available_water_std
        self.rng = np.random.default_rng(seed)

        # weather data
        self.weather_schedule = weather_schedule
        self.weather_forecast_stds = weather_forecast_stds

        # gym parameters
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float64)
        self.action_space = gym.spaces.Box(0, 1e9, shape=(1,), dtype=np.float64)
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

    def weather_info(self, day, noisy=False):
        """return a weather info for a certain day"""
        if day >= self.num_growing_days:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0])
        if not noisy:
            return np.array(
                [self.weather_schedule.max_temp[day],  # degrees C
                 self.weather_schedule.min_temp[day],  # degrees C
                 self.weather_schedule.precipitation[day],  # mm
                 self.weather_schedule.radiation[day],  # MJ/(m**2 * day)
                 self.weather_schedule.co2[day],  # PPM
                 self.weather_schedule.avg_vapor_pressure[day],  # hPa
                 self.weather_schedule.mean_temp[day],  # degrees C; not in SIMPLE model
                 self.weather_schedule.avg_wind[day]  # m/s; not in SIMPLE model
                 ]
            )
        else:
            return np.array(
                [self.rng.normal(self.weather_schedule.max_temp[day], self.weather_forecast_stds.max_temp),
                 self.rng.normal(self.weather_schedule.min_temp[day], self.weather_forecast_stds.min_temp),
                 self.rng.normal(self.weather_schedule.precipitation[day], self.weather_forecast_stds.precipitation),
                 self.rng.normal(self.weather_schedule.radiation[day], self.weather_forecast_stds.radiation),
                 self.rng.normal(self.weather_schedule.co2[day], self.weather_forecast_stds.co2),
                 self.rng.normal(self.weather_schedule.avg_vapor_pressure[day],
                                 self.weather_forecast_stds.avg_vapor_pressure),
                 self.rng.normal(self.weather_schedule.mean_temp[day], self.weather_forecast_stds.mean_temp),
                 self.rng.normal(self.weather_schedule.avg_wind[day], self.weather_forecast_stds.avg_wind)
                 ]
            )

    def create_dict_state(self):
        """
        Things to include in the state would be:
            - Actual cumulative biomass
            - Noisy reading of cumulative biomass (such as might be estimated from an image of the plant?)
            - Cumulative mean temp (seems relatively measurable)
            - Days since sowing date
            - Weather from last day
            - Next day's weather forecast (perfect)
            - Noisy "prediction" of next day's weather forecast
            - Root zone available water reading (like a soil measurement?)
            - Noisy reading of root zone available water
        """
        return {'cumulative_biomass': np.array(self.cumulative_biomass).reshape((1,)),
                'cumulative_biomass_noisy':
                    np.array(max(0, self.rng.normal(self.cumulative_biomass,
                                                    self.cumulative_biomass_sigma))).reshape((1,)),
                'cumulative_mean_temp': np.array(self.cumulative_mean_temp).reshape((1,)),
                'day': np.array(self.day).reshape((1,)),
                'weather_today': self.weather_info(self.day),
                'weather_tomorrow': self.weather_info(self.day + 1),
                'weather_tomorrow_forecast': self.weather_info(self.day + 1, noisy=True),
                'plant_available_water': np.array(self.plant_available_water).reshape((1,)),
                'plant_available_water_noisy':
                    np.array(max(0,
                                 self.rng.normal(self.plant_available_water,
                                                 self.plant_available_water_sigma))).reshape((1,))
                }

    @staticmethod
    def create_numpy_state(dict_state: dict):
        return np.concatenate([v for v in dict_state.values()])

    @staticmethod
    def create_info(dict_state: dict, additional_info: dict):
        info = copy.deepcopy(dict_state)
        info.update(additional_info)
        return info

    def irrigation_reward(self, irrigation):
        # assume irrigation is for a non-towable center pivot, and consider
        # http://h2oinitiative.com/wp-content/uploads/2018/05/Estimating-Irrigation-Costs-Tacker-et-al.pdf
        # even though it is from Arkansas...
        # 9 inch irrigation season is $116/acre ~ $286.64/hectare assuming 2.471 acres per hectare
        # 9 inches is 22.86 cm, or 0.2286 m, and 1 hectare = 10,000 m**2, so price per mm for 1 square meter plot is
        # $286.64/(10,000 m**2 * 228.6 mm) = $0.000125 / mm / m**2
        return -0.000125 * irrigation

    def step(self, action):
        rad_day = self.weather_schedule.radiation[self.day]
        mean_temp_day = self.weather_schedule.mean_temp[self.day]
        max_temp_day = self.weather_schedule.max_temp[self.day]
        min_temp_day = self.weather_schedule.min_temp[self.day]
        avg_vapor_pressure = self.weather_schedule.avg_vapor_pressure[self.day]
        avg_wind = self.weather_schedule.avg_wind[self.day]
        precipitation = self.weather_schedule.precipitation[self.day]
        irrigation = action[0]
        co2_day = self.weather_schedule.co2[self.day]

        self.cumulative_mean_temp += delta_cumulative_temp(mean_temp_day, self.crop_temp_base)
        f_heat = calc_f_heat(max_temp_day, self.crop_heat_stress_thresh, self.crop_heat_stress_extreme)
        ref_evapotranspiration = calc_ref_evapotranspiration(self.date, self.latitude, self.elevation, min_temp_day,
                                                             max_temp_day, rad_day, avg_vapor_pressure, avg_wind)
        transpiration = calc_transpiration(ref_evapotranspiration, self.plant_available_water)  # use PAW from today
        # update PAW
        self.plant_available_water = calc_plant_available_water(self.plant_available_water,
                                                                precipitation, irrigation,
                                                                transpiration,
                                                                self.crop_deep_drainage_coef,
                                                                self.crop_root_zone_depth,
                                                                self.crop_water_holding_capacity,
                                                                self.crop_runoff_curve_number)
        arid_index = calc_arid(transpiration, ref_evapotranspiration)
        f_water = calc_f_water(self.crop_drought_stress_sensitivity, arid_index)
        f_solar, senescence = calc_f_solar(self.cumulative_mean_temp, self.crop_rad_50p_growth, self.crop_maturity_temp,
                                           self.crop_rad_50p_senescence)
        if senescence:  # Note: Not sure if this is how to correctly designate growth vs senescence periods
            self.crop_rad_50p_senescence = calc_rad_50p_senescence(self.crop_rad_50p_senescence,
                                                                   self.crop_rad_50p_max_heat,
                                                                   self.crop_rad_50p_max_water,
                                                                   f_heat,
                                                                   f_water)
        f_co2 = calc_f_co2(self.crop_co2_sensitivity, co2_day)
        f_temp = calc_f_temp(mean_temp_day, self.crop_temp_base, self.crop_temp_opt)
        f_solar_water = calc_f_solar_water(f_water)
        biomass_rate = calc_biomass_rate(rad_day, f_solar, f_solar_water, self.crop_RUE, f_co2, f_temp, f_heat, f_water)
        self.cumulative_biomass = calc_cumulative_biomass(self.cumulative_biomass, biomass_rate)

        # do this before updating day to get accurate states
        dict_state = self.create_dict_state()
        state = self.create_numpy_state(dict_state)
        info = self.create_info(dict_state, {'arid_index': arid_index, 'f_solar': f_solar})

        if self.day >= self.num_growing_days - 1:
            reward = calc_yield(self.cumulative_biomass, self.crop_harvest_index) + self.irrigation_reward(irrigation)
            return state, reward, True, info

        self.day += 1
        self.date += datetime.timedelta(days=1)
        return state, self.irrigation_reward(irrigation), False, info

    def reset(self):
        self.cumulative_mean_temp = self.init_cumulative_temp  # TT variable in paper
        self.cumulative_biomass = self.init_biomass
        self.crop_rad_50p_senescence = self.init_crop_rad_50p_senescence
        # root zone available water (W_i) = root zone available water content (theta^{ad}_{a, i} or PAW) times the root
        # zone depth (RZD)
        self.plant_available_water = self.initial_plant_available_water
        self.date = self.sowing_date
        self.day = 0
        return self.create_numpy_state(self.create_dict_state())

    def render(self, mode='human'):
        pass
