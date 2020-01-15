"""Function to plot in a configuration from a generation"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os
from location import Location

def plot(simulator, calculatecost, configuration=None, model_name=None, generation_number=None, load2=True, sp_cost_per_sm=190,  sp_eff=16) :
    if(configuration is None) :
        configuration = load(model_name=model_name, generation_number=generation_number, load2 = True)
        print(configuration)
    total_power, _ = simulator.calc_total_power(configuration[0:-2], [configuration[-2], configuration[-1]], sp_eff)
    wind_power, wind_energy = simulator.calc_wind([configuration[-2], configuration[-1]])
    oppervlakte = [configuration[0],configuration[3],configuration[6],configuration[9]]
    angle = [configuration[1],configuration[4],configuration[7],configuration[10]]
    orientation = [configuration[2],configuration[5],configuration[8],configuration[11]]
    solar_power, solar_energy = simulator.calc_solar(Az=orientation, Inc=angle, sp_area=oppervlakte) #todo arrays maken voor orientatie, angle en oppervlakte
    max_power = total_power.max()
    # get the dictionary of a configuration that has the stats of the configuration
    dic = calculatecost.get_stats(total_power,sp_cost_per_sm,4,int(configuration[-2]))
    # creating an array of the mean values from the power output of the simulation
    print(sum(solar_power))
    print(sum(wind_power))
    wind_power = np.mean(np.reshape(wind_power[:8760], (365,24)),axis=1)
    solar_power = np.mean(np.reshape(solar_power[:8760], (365,24)),axis=1)
    consumption = np.full(len(total_power), 6000)
    return consumption, total_power, solar_power, wind_power, dic, configuration, max_power

def draw_energy(consumption, total_power, solar_power, wind_power, dic, configuration, max_power):
    # Creating mutiple text variables to display in the graph
    total_power = np.mean(np.reshape(total_power[:8760], (365,24)), axis=1)
    power_generated = total_power.sum()
    t1 = "Storage capacity: \nAmount of windturbines: \nCable area: \nMaximum Power Output: \nTotal Power Generated: \nTotal costs: "
    t2 = str(int(dic['total_storage'])) + " kWh\n" + \
        str(int(configuration[-2])) + "\n" + \
        str(int(dic['cable_area'])) + " mm²\n" + \
        str(int(max_power)) + " kW\n" + \
        str(int(power_generated)) + " kWh\n" +\
        '€' + str(int(dic['cost']))
    # Creating the solar stats text variables to display in the graph
    t3 = ""
    for I in range(4):
        if configuration[0 + I*3] > 0:
            t3 = t3 + "SP" + str(I + 1) + " - Area: " + str(int(configuration[0 + I*3])) +\
                "m² - Angle: " + str(int(configuration[1 + I*3])) +\
                "° - Orientation: " + str(int(configuration[2 + I*3])) + "°\n"
    plt.subplot(2, 1, 1)
    plt.plot(total_power, color='green', alpha=0.5, label='Total energy production')
    plt.plot(solar_power, color='yellow', alpha=0.5, label='Solar energy')
    plt.plot(wind_power, color='blue', alpha=0.5, label='Wind energy')
    plt.plot(consumption, color='red', label='Energy demand')
    plt.text(330, total_power.max() * 1.04, t2, ha='left', va='top', style='italic', wrap=False)
    plt.text(330, total_power.max() * 1.04, t1, ha='right', va='top', wrap=False)
    plt.text(362, total_power.max() * 0.725, t3, ha='right', va='top', wrap=False)
    plt.legend(loc='upper center')
    plt.title("Power Average per Day")
    plt.xlabel('Days')
    plt.ylabel('kW')
    plt.xlim(0,365)
    plt.subplot(2, 1, 2)
    plt.plot(np.cumsum(total_power - 6000), color='green')
    plt.xlim(0,365)
    plt.title("Energy balance")
    plt.xlabel('Days')
    plt.ylabel('kWh')
    plt.show()

def draw_Battery_Use(consumption, total_power, solar_power, wind_power, dic, configuration, max_power):
    # Creating mutiple text variables to display in the graph
    power_generated = total_power.sum()
    power = total_power
    total_power = np.mean(np.reshape(total_power[:8760], (365,24)), axis=1)
    t1 = "Storage capacity: \nAmount of windturbines: \nCable area: \nMaximum Power Output: \nTotal Power Generated: \nTotal costs: "
    t2 = str(int(dic['total_storage'])) + " kWh\n" + \
        str(int(configuration[-2])) + "\n" + \
        str(int(dic['cable_area'])) + " mm²\n" + \
        str(int(max_power)) + " kW\n" + \
        str(int(power_generated)) + " kWh\n" +\
        '€' + str(int(dic['cost']))
    # Creating the solar stats text variables to display in the graph
    t3 = ""
    for I in range(4):
        if configuration[0 + I*3] > 0:
            t3 = t3 + "SP" + str(I + 1) + " - Area: " + str(int(configuration[0 + I*3])) +\
                "m² - Angle: " + str(int(configuration[1 + I*3])) +\
                "° - Orientation: " + str(int(configuration[2 + I*3])) + "°\n"

    plt.subplot(2, 1, 1)
    plt.plot(total_power, color='green', alpha=0.5, label='Total energy production')
    plt.plot(solar_power, color='yellow', alpha=0.5, label='Solar energy')
    plt.plot(wind_power, color='blue', alpha=0.5, label='Wind energy')
    plt.plot(consumption, color='red', label='Energy demand')
    plt.text(330, total_power.max() * 1.04, t2, ha='left', va='top', style='italic', wrap=False)
    plt.text(330, total_power.max() * 1.04, t1, ha='right', va='top', wrap=False)
    plt.text(362, total_power.max() * 0.725, t3, ha='right', va='top', wrap=False)
    plt.legend(loc='upper center')
    plt.title("Power Average per Day")
    plt.xlabel('Days')
    plt.ylabel('kW')
    plt.xlim(0,365)
    plt.subplot(2, 1, 2)
    power = power - 6000
    for x in range(2) :
        if x == 0 :
            batterycharge = [int(dic['total_storage'])]
        else:
            batterycharge = [batterycharge[-1]]
        Powershortage = []
        for I in power :
            batterycharge.append(batterycharge[-1] + I)
            if(int(dic['total_storage']) < batterycharge[-1]) : 
                batterycharge[-1] = int(dic['total_storage'])
            elif(0 > batterycharge[-1]) :
                batterycharge[-1] = 0
                Powershortage.append(len(batterycharge)-1)
    plt.plot(batterycharge, color='green', alpha=0.5)
    if len(Powershortage) == 0:
        plt.scatter(np.zeros(len(Powershortage)), Powershortage, color='red')
    plt.title("Power supply level over a Year")
    plt.xlabel('Hour')
    plt.ylabel('kWh')
    plt.xlim(0,8760)
    plt.show()

"""def plot_solarenergy(generation):
    npempty = np.zeros(9)
    colors = np.array(['green','blue','red','yellow'])
    text = ""
    for I in range(4):
        if(generation[0 + I*3]>0):
            solar_distribution, _ = simulink.run_simulation(np.concatenate((generation[0 + I*3:3 + I*3], npempty), axis=None), 4, 0)
            text = "SP" + str(I + 1) + " - Area: " + str(int(generation[0 + I*3])) +\
                "m² - Angle: " + str(int(generation[1 + I*3])) +\
                "° - Orientation: " + str(int(generation[2 + I*3])) + "°\n"
            plt.plot(np.mean(np.reshape(solar_distribution[:8760], (365,24)), axis=1), color=colors[I], alpha=0.5, label=text)
    plt.legend()
    plt.show()"""    

def load(model_name, generation_number, takebest=True, load2=True):
    if model_name is None or generation_number is None:
        raise Exception('None attribute detected on model or configuration parameter')
    elif generation_number < 0:
        raise Exception('There can be no configuration with a number less then zero')
    if load2:
        path = 'saved_runs'+ os.sep + model_name + os.sep
    else:
        path = model_name
    if takebest: 
        return np.loadtxt(path + 'best_' + str(generation_number) + '.csv', delimiter=',')[0]
    else:
        return np.loadtxt(path + 'generation_' + str(generation_number) + '.csv', delimiter=',')[0]
    

if __name__ == '__main__':
    from generators import Windturbine
    from calculate_cost import CostCalculator
    from Simulator import Simulator

    # table of the diffrent cable cost
    turbine = Windturbine(4)
    calculatecost = CostCalculator(160, 400, 6000, 1000000, 1000)
    simulate = Simulator(Location('Volkel'),"2018",Windturbine(4))
    consumption, total_power, solar_power, wind_power, dic, configuration, max_power = plot(simulate, calculatecost, model_name='20200107_112010', generation_number=99)
    draw_energy(consumption, total_power, solar_power, wind_power, dic, configuration, max_power)
