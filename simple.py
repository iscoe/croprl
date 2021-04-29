import numpy as np
from pcse.util import penman_monteith
import datetime
import gym

T_BASE = 0
LAT = 43.9271  # latitude of Hamer, ID; degrees N, used for ET0 computation
ELEV = 1464  # elevation of Hamer, ID in meters above sea level
WIND2 = 10.299802  # average wind speed in Hamer, ID in m/s (at 2 meter?)
VAP = 4.04  # vapor pressure in Hamer ID on 24 Apr 2021 based on a temp of 54F and dewpoint of 22, and according to
#   https://www.weather.gov/epz/wxcalc_vaporpressure
F_SOLAR_MAX = 0.95  # the maximum fraction of radiation interception that a crop can reach, governed by plant spacings,


#   but typically set to 0.95 according to SIMPLE model paper:
#   https://www.sciencedirect.com/science/article/pii/S1161030118304234#bib0205


def dq_d_temp(temp):
    """
    derivative of specific humidity/saturated vapour pressure at a given temperature
    See:
        - https://journals.ametsoc.org/view/journals/mwre/100/2/1520-0493_1972_100_0081_otaosh_2_3_co_2.xml?tab_body=pdf
        - A COMPARISONOF THE PRIESTLEY-TAYLOR AND PENMAN METHODS FOR ESTIMATING REFERENCE CROP EVAPOTRANSPIRATION IN
            TROPICAL COUNTRIES, H. GUNSTON and C.H. BATCHELOR
        - https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml

        P_s = a * e^( bt / (t + c) )
            where a = 610.94, b = 17.625, and c = 243.04

        so d(P_s)/dt = (b * t / (t + c)) * ((b * (t + c) - b * c* t) / ((t + c) ** 2) * a * e^( bt / (t + c) )
    """
    a, b, c = 610.94, 17.625, 243.04
    front = (b * temp / (temp + c)) * ((b * (temp + c) - b * c * temp) / ((temp + c) ** 2) * a)
    return front * np.exp(b * temp / (temp + c))


def calc_plant_available_water(rzaw_day_before, precipitation, irrigation, transpiration, deep_drainage_coef,
                               root_zone_depth, water_holding_capacity, runoff_curve_number):
    # Note: Plant available water == Available water content (AWC)
    # T_i = min(alpha * zeta * theta^(ad)_{a, i-1}, ET0_i) = min(alpha * PAW_{i-1}, ET0_i)
    # ~ min(alpha * W, ET0_i)
    # PAW_i = theta^(ad)_{a, i-1} = W_i / zeta
    # W_i = W_{i-1} + P_i + I_i - T_i - D_i - R_i
    # P_i: precipitation on day i in mm
    # I_i: irrigation on day i in mm
    # T_i: transpiration
    # D_i: deep drainage
    # R_i: surface runoff
    # https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286

    # get available water before runoff and drainage
    root_zone_available_water = rzaw_day_before + precipitation + irrigation - transpiration
    deep_drainage = max(0,
                        (
                                deep_drainage_coef * root_zone_depth) *
                        (root_zone_available_water / root_zone_depth - water_holding_capacity)
                        )
    potential_maximum_retention = 25400 / runoff_curve_number - 254
    initial_abstraction = 0.2 * potential_maximum_retention
    if precipitation > initial_abstraction:
        surface_runoff = ((precipitation - initial_abstraction) * (precipitation - initial_abstraction) /
                          (precipitation - initial_abstraction + potential_maximum_retention))
    else:
        surface_runoff = 0
    root_zone_available_water = root_zone_available_water - deep_drainage - surface_runoff
    return root_zone_available_water / root_zone_depth, root_zone_available_water


def calc_transpiration(ref_evapotranspiration, plant_available_water):
    return min(0.096 * plant_available_water, ref_evapotranspiration)


def delta_cumulative_temp(temp, temp_base):
    return temp - temp_base if temp > temp_base else 0


def calc_biomass_rate(radiation, f_solar, rue, f_co2, f_temp, f_heat, f_water):
    return radiation * f_solar * rue * f_co2 * f_temp * min(f_heat, f_water)


def calc_cumulative_biomass(biomass_day_before, biomass_rate):
    return biomass_day_before + biomass_rate


def calc_yield(end_cumulative_biomass, harvest_index):
    return end_cumulative_biomass * harvest_index


def calc_f_solar(cumulative_mean_temp, rad_50p_growth, maturity_temp, rad_50p_senescence):
    # assumes maturity or senescence after cumulative mean temperature is grater than maturity_temp.
    if cumulative_mean_temp >= maturity_temp:
        return F_SOLAR_MAX / (1 + np.exp(0.01 * (cumulative_mean_temp - (maturity_temp - rad_50p_senescence))))
    else:
        return F_SOLAR_MAX / (1 + np.exp(-0.01 * (cumulative_mean_temp - rad_50p_growth)))


def calc_f_temp(temp, temp_base, temp_opt):
    if temp < temp_base:
        return 0
    elif temp < temp_opt:
        return (temp - temp_base) / (temp_opt - temp_base)
    else:
        return 1


def calc_f_heat(max_temp_day, crop_heat_stress_thresh, crop_extreme_heat_stress_thresh):
    if max_temp_day <= crop_heat_stress_thresh:
        return 1
    elif max_temp_day <= crop_extreme_heat_stress_thresh:
        return 1 - ((max_temp_day - crop_heat_stress_thresh)
                    / (crop_extreme_heat_stress_thresh - crop_heat_stress_thresh))
    else:
        return 0


def calc_f_co2(s_co2, co2):
    if 350 <= co2 <= 700:
        return 1 + s_co2 * (co2 - 350)
    elif co2 > 700:
        return 1 + s_co2 * 350
    else:
        raise ValueError("Invalid CO2 value: {}".format(co2))


def calc_f_water(s_water, arid):
    return 1 - s_water * arid


def calc_arid(transpiration, ref_evapotranspiration):
    return 1 - transpiration / ref_evapotranspiration


def calc_rad_50p_senescence(rad_50p_before, rad_50p_max_heat, rad_50p_max_water, f_heat, f_water):
    # I_{50B} term
    return rad_50p_before + rad_50p_max_heat * (1 - f_heat) + rad_50p_max_water * (1 - f_water)


def calc_f_solar_water(f_water):
    return 0.9 + f_water if f_water < 0.1 else 1.0


class RandomWeatherSchedule:
    """should subclass from some type of WeatherSchedule object that validates settings?"""

    def __init__(self, num_days):
        # all made up data, loosely based on starting May 1st in Hamer, ID
        self.num_days = num_days
        rng = np.random.default_rng()
        max_temp = rng.normal(15, 3, size=num_days)  # degrees C
        min_temp = rng.normal(2, 2, size=num_days)  # degrees C
        mean_temp = rng.normal((max_temp - min_temp) / 2, 2, size=num_days)  # degrees C
        self.__dict__.update({
            'max_temp': max_temp,  # degrees C
            'min_temp': min_temp,  # degrees C
            'precipitation': np.maximum(0, rng.normal(0, 3, size=num_days)),  # mm
            'radiation': np.maximum(0, rng.normal(22, 1, size=num_days)),  # MJ/(m**2 * day)
            'co2': rng.normal(412, 1, size=num_days),  # PPM
            'avg_vapor_pressure': np.maximum(0, rng.normal(4.04, 1, size=num_days)),  # hPa
            'mean_temp': mean_temp,  # degrees C; not in SIMPLE model
            'avg_wind': np.maximum(0, rng.normal(8, 6, size=num_days))  # m/s; not in SIMPLE model
        })


class WeatherForecastSTDs:
    def __init__(self):
        self.__dict__.update({
            'max_temp': 3,  # degrees C
            'min_temp': 2,  # degrees C
            'precipitation': 3,  # mm
            'radiation': 1,  # MJ/(m**2 * day)
            'co2': 1,  # PPM
            'avg_vapor_pressure': 1,  # hPa
            'mean_temp': 2,  # degrees C; not in SIMPLE model
            'avg_wind': 6  # m/s; not in SIMPLE model
        })


class PotatoRussetUSACropParametersSpec:
    """should subclass from some type of CropSpec object that validates settings?"""

    def __init__(self):
        # crop parameters
        self.temp_base = 4  # T_base
        self.temp_opt = 22  # T_opt
        self.RUE = 1.30  # RUE
        self.rad_50p_growth = 500  # I_50A
        self.rad_50p_senescence = 400  # I_50B
        self.maturity_temp = 2300  # T_sum
        self.rad_50p_max_heat = 50  # I_50maxH
        self.rad_50p_max_water = 30  # I_50maxW
        self.heat_stress_thresh = 34  # T_max
        self.heat_stress_extreme = 45  # T_ext
        self.drought_stress_sensitivity = 0.4  # S_water
        self.deep_drainage_coef = 0.8  # DDC
        self.water_holding_capacity = 0.1  # AWC
        self.runoff_curve_number = 64  # RCN
        self.root_zone_depth = 1200  # RZD
        self.co2_sensitivity = 0.10  # S_CO2
        self.harvest_index = 0.9  # HI


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
                 initial_root_zone_available_water_content=None,  # not a parameter in SIMPLE, but reasonable to vary,
                 #  https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286 assumes initial value to
                 #  be equal to the available water capacity (AWC) in their crop model comparison
                 seed=None,
                 cumulative_biomass_std=1.0,
                 root_zone_available_water_std=1.0):

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
        self.init_biomass = initial_biomass
        self.sowing_date = sowing_date

        # root zone available water (W_i) = root zone available water content (theta^{ad}_{a, i} or PAW) times the root
        #   zone depth (RZD), i.e. W_i =  theta^{ad}_{a, i} * RZD. See:
        #   https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286, which also justifies the initial
        #   condition shown here in the crop comparison section
        if initial_root_zone_available_water_content is None:
            initial_root_zone_available_water_content = self.crop_water_holding_capacity
        self.initial_root_zone_available_water_content = initial_root_zone_available_water_content
        self.initial_root_zone_available_water = initial_root_zone_available_water_content * self.crop_root_zone_depth

        # tracking variables
        self.cumulative_mean_temp = None  # TT variable in paper
        self.cumulative_biomass = None
        self.root_zone_available_water = None
        self.plant_available_water = None
        self.date = None
        self.day = None

        # Randomization parameters
        self.cumulative_biomass_sigma = cumulative_biomass_std
        self.root_zone_available_water_sigma = root_zone_available_water_std
        self.rng = np.random.default_rng(seed)

        # weather data
        self.weather_schedule = weather_schedule
        self.weather_forecast_stds = weather_forecast_stds

    def weather_info(self, day, noisy=False):
        """return a weather info for a certain day"""
        if day >= self.num_growing_days:
            return np.array([])
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

    def create_state(self):
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
        return {'cumulative_biomass': self.cumulative_biomass,
                'cumulative_biomass_noisy': max(0, self.rng.normal(self.cumulative_biomass,
                                                                   self.cumulative_biomass_sigma)),
                'cumulative_mean_temp': self.cumulative_mean_temp,
                'day': self.day,
                'weather_today': self.weather_info(self.day),
                'weather_tomorrow': self.weather_info(self.day + 1),
                'weather_tomorrow_forecast': self.weather_info(self.day + 1, noisy=True),
                'root_zone_available_water': self.root_zone_available_water,
                'root_zone_available_water_noisy': max(0, self.rng.normal(self.root_zone_available_water,
                                                                          self.root_zone_available_water_sigma))
                }

    def step(self, action):
        rad_day = self.weather_schedule.radiation[self.day]
        mean_temp_day = self.weather_schedule.mean_temp[self.day]
        max_temp_day = self.weather_schedule.max_temp[self.day]
        min_temp_day = self.weather_schedule.min_temp[self.day]
        avg_vapor_pressure = self.weather_schedule.avg_vapor_pressure[self.day]
        avg_wind = self.weather_schedule.avg_wind[self.day]
        precipitation = self.weather_schedule.precipitation[self.day]
        irrigation = action
        co2_day = self.weather_schedule.co2[self.day]

        self.cumulative_mean_temp += delta_cumulative_temp(mean_temp_day, self.crop_temp_base)
        f_heat = calc_f_heat(max_temp_day, self.crop_heat_stress_thresh, self.crop_heat_stress_extreme)
        # penman_monteith function has rads in Joules, while rest of SIMPLE uses megajoules:
        # https://github.com/ajwdewit/pcse/blob/c40362be6a176dabe42a39b4526015b41cf23c48/pcse/util.py#L129
        rad_day_joules = 1e6 * rad_day
        ref_evapotranspiration = penman_monteith(self.date, self.latitude, self.elevation, min_temp_day, max_temp_day,
                                                 rad_day_joules, avg_vapor_pressure, avg_wind)
        transpiration = calc_transpiration(ref_evapotranspiration, self.plant_available_water)  # use PAW from today
        # update PAW
        self.plant_available_water, self.root_zone_available_water \
            = calc_plant_available_water(self.root_zone_available_water,
                                         precipitation, irrigation,
                                         transpiration,
                                         self.crop_deep_drainage_coef,
                                         self.crop_root_zone_depth,
                                         self.crop_water_holding_capacity,
                                         self.crop_runoff_curve_number)
        # todo: Validate ARID values
        arid_index = calc_arid(transpiration, ref_evapotranspiration)
        f_water = calc_f_water(self.crop_drought_stress_sensitivity, arid_index)
        self.crop_rad_50p_senescence = calc_rad_50p_senescence(self.crop_rad_50p_senescence, self.crop_rad_50p_max_heat,
                                                               self.crop_rad_50p_max_water, f_heat, f_water)
        f_solar = calc_f_solar(self.cumulative_mean_temp, self.crop_rad_50p_growth, self.crop_maturity_temp,
                               self.crop_rad_50p_senescence)
        f_co2 = calc_f_co2(self.crop_co2_sensitivity, co2_day)
        f_temp = calc_f_temp(mean_temp_day, self.crop_temp_base, self.crop_temp_opt)
        biomass_rate = calc_biomass_rate(rad_day, f_solar, self.crop_RUE, f_co2, f_temp, f_heat, f_water)
        self.cumulative_biomass = calc_cumulative_biomass(self.cumulative_biomass, biomass_rate)

        # do this before updating day to get accurate states
        state = self.create_state()

        if self.day >= self.num_growing_days - 1:
            return state, calc_yield(self.cumulative_biomass, self.crop_harvest_index), True, {}

        self.day += 1
        self.date += datetime.timedelta(days=1)
        return state, 0, False, {}

    def reset(self):
        self.cumulative_mean_temp = self.init_cumulative_temp  # TT variable in paper
        self.cumulative_biomass = self.init_biomass
        # root zone available water (W_i) = root zone available water content (theta^{ad}_{a, i} or PAW) times the root
        # zone depth (RZD)
        self.root_zone_available_water = self.initial_root_zone_available_water
        self.plant_available_water = self.initial_root_zone_available_water_content
        self.date = self.sowing_date
        self.day = 0
        return self.create_state()

    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    day = datetime.datetime.today()
    temp_min = 23
    temp_max = 25  # made up
    avg_rad = 4
    ET0 = penman_monteith(day, LAT, ELEV, temp_min, temp_max, avg_rad, VAP, WIND2)
    T_i = calc_transpiration(ET0, 1)
    # US potato constants
    awc = 0.1
    rcn = 64
    ddc = 0.8
    rzd = 1200
    paw = calc_plant_available_water(0, 5, 2, T_i, ddc, rzd, awc, rcn)
    ARID = calc_arid(T_i, ET0)

    print(ARID)

    growth_days = 10
    start_date = datetime.datetime(day=1, month=5, year=2021)
    weather = RandomWeatherSchedule(growth_days)
    weather_stds = WeatherForecastSTDs()
    crop_params = PotatoRussetUSACropParametersSpec()
    env = SimpleCropModelEnv(start_date, growth_days, weather, WeatherForecastSTDs(), LAT, ELEV, crop_params, seed=0)
    env.reset()
    done = False
    iter = 0
    while not done and iter < 1000000:
        action = 0.1
        s, r, done, info = env.step(action)
        print(s)
        print("reward: ", r)
        iter += 1
