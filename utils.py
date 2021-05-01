import numpy as np

hamer_avg_lows = {"May": 2, "Jun": 6, "Jul": 9, "Aug": 8, "Sep": 3, "Oct": -3}
hamer_avg_highs = {"May": 21, "Jun": 26, "Jul": 31, "Aug": 30, "Sep": 24, "Oct": 16}
hamer_avg_precip = {"May": 33.5, "Jun": 28.4, "Jul": 17.1, "Aug": 15.3, "Sep": 14.8, "Oct": 18}


def grams_per_square_meter_to_tons_per_hectare(biomass):
    # 1 hectare = 10,000 m**2
    # 1 metric ton = 1000 kg
    # See https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GB002947, which is ref'd in SIMPLE paper
    # new_value = biomass * 10000 / (1000*1000) = biomass / 100
    return biomass / 100
