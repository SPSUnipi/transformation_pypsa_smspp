# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:59:51 2025

@author: aless
"""


import pypsa
import pandas as pd
import re
import numpy as np


def clean_marginal_cost(n):
    n.links.marginal_cost = 0
    n.storage_units.marginal_cost = 0
    n.stores.marginal_cost = 0
    return n
    
def clean_global_constraints(n):
    n.global_constraints = pd.DataFrame(columns=n.global_constraints.columns)
    n.global_constraints_t = dict()
    return n
    
def clean_e_sum(n):
    n.generators.e_sum_max = float('inf')
    n.generators_e_sum_min = float('-inf')
    return n

def clean_efficiency_link(n):
    n.links.efficiency = 1
    return n

def clean_ciclicity_storage(n):
    n.storage_units.cyclic_state_of_charge = False
    n.storage_units.cyclic_state_of_charge_per_period = False
    n.storage_units.state_of_charge_initial = n.storage_units.max_hours * n.storage_units.p_nom
    return n

def clean_marginal_cost_intermittent(n):
    renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'PV', 'wind', 'ror']
    renewable_mask = n.generators.carrier.isin(renewable_carriers)
    n.generators.loc[renewable_mask, 'marginal_cost'] = 0.0
    return n

def clean_storage_units(n):
    n.storage_units.drop(n.storage_units.index, inplace=True)
    for key in n.storage_units_t.keys():
        n.storage_units_t[key].drop(columns=n.storage_units_t[key].columns, inplace=True)
    return n

def clean_stores(n):
    n.stores.drop(n.stores.index, inplace=True)
    for key in n.stores_t.keys():
        n.stores_t[key].drop(columns=n.stores_t[key].columns, inplace=True)
    return n

    

def parse_txt_file(file_path):
    data = {'DCNetworkBlock': {'PowerFlow': []}}
    current_block = None  

    with open(file_path, "r") as file:
        for line in file:
            match_time = re.search(r"Elapsed time:\s*([\deE\+\.-]+)\s*s", line)
            if match_time:
               elapsed_time = float(match_time.group(1))
               data['elapsed_time'] = elapsed_time
               break  
            
            block_match = re.search(r"(ThermalUnitBlock|BatteryUnitBlock|IntermittentUnitBlock|DCNetworkBlock)", line)
            if block_match:
                current_block = block_match.group(1)
                if current_block != 'DCNetworkBlock':
                    data[current_block] = {}  
                continue  

            match = re.match(r"([\w\s]+)\s+=\s+\[([^\]]*)\]", line)
            if match and current_block:
                key, values = match.groups()
                key = key.strip()  

                # Se è DCNetworkBlock, accumula in una lista temporanea
                if current_block == 'DCNetworkBlock':
                    data[current_block]['PowerFlow'].extend([float(x) for x in values.split()])
                else:
                    data[current_block][key] = np.array([float(x) for x in values.split()])  

    # Converti PowerFlow in un array NumPy alla fine
    if data['DCNetworkBlock']['PowerFlow']:
        data['DCNetworkBlock']['PowerFlow'] = np.array(data['DCNetworkBlock']['PowerFlow'])

    return data

#%% Network definition with PyPSA

if __name__ == '__main__':
    network_name = "base_s_5_elec_lvopt_1h"
    network = pypsa.Network(f"../test/networks/{network_name}.nc")
    
    network = clean_marginal_cost(network)
    network = clean_global_constraints(network)
    network = clean_e_sum(network)



