# Copyright 2020-2021, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the Apache 2.0 License.

import numpy as np
from pcse.util import penman_monteith

"""
Implementations of the functions given in the SIMPLE paper:?

https://www.sciencedirect.com/science/article/pii/S1161030118304234

used to step the crop model.
"""

F_SOLAR_MAX = 0.95  # the maximum fraction of radiation interception that a crop can reach, governed by plant spacings,


#                       but typically set to 0.95 according to SIMPLE model paper:
#                       https://www.sciencedirect.com/science/article/pii/S1161030118304234#bib0205


def calc_plant_available_water(paw_day_before, precipitation, irrigation, transpiration, deep_drainage_coef,
                               root_zone_depth, water_holding_capacity, runoff_curve_number):
    # Note: Plant available water == Available water content (AWC)
    # T_i = min(alpha * zeta * theta^(ad)_{a, i-1}, ET0_i) = min(alpha * PAW_{i-1}, ET0_i)
    # ~ min(alpha * W, ET0_i)
    # PAW_i = zeta * theta^(ad)_{a, i-1} = zeta * (W_i / zeta) = W_i
    # W_i = W_{i-1} + P_i + I_i - T_i - D_i - R_i
    # P_i: precipitation on day i in mm
    # I_i: irrigation on day i in mm
    # T_i: transpiration
    # D_i: deep drainage
    # R_i: surface runoff
    # https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286

    # get available water before runoff and drainage
    plant_available_water = paw_day_before + precipitation + irrigation - transpiration
    deep_drainage = max(0,
                        (
                                deep_drainage_coef * root_zone_depth) *
                        (plant_available_water / root_zone_depth - water_holding_capacity)
                        )
    potential_maximum_retention = 25400 / runoff_curve_number - 254
    initial_abstraction = 0.2 * potential_maximum_retention
    if precipitation > initial_abstraction:
        surface_runoff = ((precipitation - initial_abstraction) * (precipitation - initial_abstraction) /
                          (precipitation - initial_abstraction + potential_maximum_retention))
    else:
        surface_runoff = 0
    plant_available_water = plant_available_water - deep_drainage - surface_runoff
    # PAW_i = zeta * theta^(ad)_{a, i-1} = zeta * (W_i / zeta) = W_i; units are mm/day
    return plant_available_water


def calc_ref_evapotranspiration(date, latitude, elevation, min_temp_day, max_temp_day, rad_day, avg_vapor_pressure,
                                avg_wind):
    # penman_monteith function has rads in Joules, while rest of SIMPLE uses megajoules:
    # https://github.com/ajwdewit/pcse/blob/c40362be6a176dabe42a39b4526015b41cf23c48/pcse/util.py#L129
    rad_day_joules = rad_day * 1e6
    return penman_monteith(date, latitude, elevation, min_temp_day, max_temp_day, rad_day_joules, avg_vapor_pressure,
                           avg_wind)


def calc_transpiration(ref_evapotranspiration, plant_available_water):
    return min(0.096 * plant_available_water, ref_evapotranspiration)  # mm/day


def delta_cumulative_temp(temp, temp_base):
    return temp - temp_base if temp > temp_base else 0  # degrees C?


def calc_biomass_rate(radiation, f_solar, f_solar_water, rue, f_co2, f_temp, f_heat, f_water):
    # unit guess: "f" values are unit-less, then
    # -radiation: (MJ/(m**2 * day))
    # -RUE: (g/(MJ * m**2))
    # -> units out are  = (MJ/(m**2 * day)) * (g/(MJ * m**2)) = g/(day m**4) ? guessing should be g/(day m**2)?

    # Note: I don't know if f_solar_water is used correctly here, but it is my best guess given the reading of what it
    #   is meant to represent
    return radiation * f_solar * f_solar_water * rue * f_co2 * f_temp * min(f_heat, f_water)


def calc_cumulative_biomass(biomass_day_before, biomass_rate):
    return biomass_day_before + biomass_rate


def calc_yield(end_cumulative_biomass, harvest_index):
    # yield is in units of metric tons per hectare in paper
    return end_cumulative_biomass * harvest_index


def calc_f_solar(cumulative_mean_temp, rad_50p_growth, maturity_temp, rad_50p_senescence, growth_term_thresh=1e-3):
    """
    Returns: f_solar, senescence (float, bool) The f_solar value and if the plant is past the growth stage
    """
    growth_e_term = np.exp(-0.01 * (cumulative_mean_temp - rad_50p_growth))
    # Note: Models when the plant has finished growth and has moved towards senescence. How this should be decided was
    #  not explicitly mentioned in the paper, so this is my best interpretation based on examining the behavior of the
    #  equations and the intended interpretation. 1e-3 was calibrated based on observation with Russet potatoes
    if growth_e_term < growth_term_thresh:
        return F_SOLAR_MAX / (1 + np.exp(0.01 * (cumulative_mean_temp - (maturity_temp - rad_50p_senescence)))), True
    else:
        return F_SOLAR_MAX / (1 + growth_e_term), False


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
    if ref_evapotranspiration > 0:
        return 1 - transpiration / ref_evapotranspiration
    else:
        # case isn't dealt with in https://www.sciencedirect.com/science/article/pii/S1161030118304234 or
        # https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286 that I know of, but this seems
        # reasonable given transpiration will always be 0 if ref_evapotranspiration is 0, meaning no transpiration was
        # possible that day (e.g. not enough sun/wind).
        return 1


def calc_rad_50p_senescence(rad_50p_before, rad_50p_max_heat, rad_50p_max_water, f_heat, f_water):
    # I_{50B} term
    return rad_50p_before + rad_50p_max_heat * (1 - f_heat) + rad_50p_max_water * (1 - f_water)


def calc_f_solar_water(f_water):
    return 0.9 + f_water if f_water < 0.1 else 1.0
