from train import train
import numpy as np
import pandas as pd
import os
from calculate_cost import CostCalculator
from genetic_algorith import GeneticAlgorith
from save_and_load import PopulationSaver
from multiprocessing import Process, Value
from generators import Windturbine
from Simulator import Simulator
from location import Location
import sys

TURBINETYPE = 4

energy_demand = 6000
mutationrate = 50
def_cost = 1000000

# solar_costs  = np.array([160, 160, 160])
# storage_cost = np.array([400, 400, 400])
# numb_of_turb = np.array([10, 10, 10])
# terrain_arr = np.array([0.12, 0.19, 0.12])
# sol_min = np.array([1000, 1000, 1000])
# sol_max = np.array([10000000, 10000000, 10000000])
loc_array = np.array(['DEBILT', 'SOESTERBERG', 
                      'STAVOREN', 'LELYSTAD', 'LEEUWARDEN', 'MARKNESSE', 'DEELEN', 'LAUWERSOOG',
                      'HEINO', 'HOOGEVEEN', 'EELDE', 'HUPSEL', 'NIEUWBEERTA' , 'TWENTHE', 'VLISSINGEN',
                      'WESTDORPE', 'WILHELMINADORP', 'HOEKVANHOLLAND', 'ROTTERDAM', 'CABAUW', 'GILZERIJEN',
                      'HERWIJNEN', 'EINDHOVEN', 'VOLKEL', 'ELL', 'MAASTRICHT', 'ARCEN'])

turbine = Windturbine(TURBINETYPE)

# Loop locations here
def collectyeardata(year):
    all_stats = pd.DataFrame(columns=['Name','Year','Lat','Lon','cost','solar_cost','wind_cost','cable_cost','cable_area','storage_cost','deficit_cost','total_surplus',
                            'total_storage','storage_st_cost','zp_st_cost','wt_st_cost','Zp1_angle','Zp1_or','Zp1_area','Zp2_angle','Zp2_or','Zp2_area','Zp3_angle',
                            'Zp3_or','Zp3_area','Zp4_angle','Zp4_or','Zp4_area','ZP_tot_area','Zp_tot_power','Turbine_n','Turbine_h','Terrain_f','Turbine_tot_power','Total_power'])
    for i in range(len(loc_array)):
        loc_data = Location(loc_array[i])
        file_name = 'Data' + os.sep + 'location_' + str(loc_data.stn) + '.xlsx'
        excel_file = pd.ExcelFile(file_name)
        years = np.array(excel_file.sheet_names)
        if str(year) in years :
            sim = Simulator(file_name, str(year), turbine, index_col=0, latitude=loc_data.latitude, longitude=loc_data.longitude, terrain_factor=loc_data.terrain)
            
            configuratie = np.array([[10000,0,0,0],[0,0,0,0],[37,0,0,0]])

            sol_power,_ = sim.calc_solar(Az=[0,0,0,0] ,Inc=[37,0,0,0] ,sp_area=[10000,0,0,0])
            sol_power_total = np.sum(sol_power)
            
            inputs = {'Name': loc_data.name,
                    'Year': year,
                    'Lat': loc_data.latitude,
                    'Lon': loc_data.longitude,
                    'storage_st_cost': 0,
                    'zp_st_cost': 0,
                    'wt_st_cost': 0,
                    'Zp1_angle': configuratie[2][0],
                    'Zp1_or': configuratie[1][0],
                    'Zp1_area': configuratie[0][0],
                    'Zp2_angle': configuratie[2][1],
                    'Zp2_or': configuratie[1][1],
                    'Zp2_area': configuratie[0][1],
                    'Zp3_angle': configuratie[2][2],
                    'Zp3_or': configuratie[1][2],
                    'Zp3_area': configuratie[0][2],
                    'Zp4_angle': configuratie[2][3],
                    'Zp4_or': configuratie[1][3],
                    'Zp4_area': configuratie[0][3],
                    'ZP_tot_area': 10000,
                    'Zp_tot_power': sol_power_total,
                    'Turbine_n': 0,
                    'Turbine_h': 0,
                    'Terrain_f': loc_data.terrain,
                    'Turbine_tot_power': 0,
                    'Total_power': 0}

            all_stats = all_stats.append(inputs, ignore_index=True)
            #Uncomment this to store all the best stats from training
            print('Done with: ' + loc_array[i] + '. Year: ' + str(year))
    all_stats.to_excel('Output_data' + os.sep + 'solardata_'+ str(year) + '_' + '.xlsx')

if __name__ == '__main__':
    collectyeardata(2018)