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
    # n.stores.marginal_cost = 0
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

def clean_p_min_pu(n):
    n.storage_units.p_min_pu = 0
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


def one_bus_network(n):
    # Delete lines
    n.lines.drop(n.lines.index, inplace=True)
    for key in n.lines_t.keys():
        n.lines_t[key].drop(columns=n.lines_t[key].columns, inplace=True)
        
    # Delete links
    n.links.drop(n.links.index, inplace=True)
    for key in n.links_t.keys():
        n.links_t[key].drop(columns=n.links_t[key].columns, inplace=True)
        
    
    n.buses = n.buses.iloc[[0]]
    n.loads = n.loads.iloc[[0]]

    n.generators['bus'] = n.buses.index[0]
    n.storage_units['bus'] = n.buses.index[0]
    n.stores['bus'] = n.buses.index[0]
    n.loads['bus'] = n.buses.index[0]
    
    n.loads_t.p_set = pd.DataFrame(n.loads_t.p_set.sum(axis=1), index=n.loads_t.p_set.index, columns=[n.buses.index[0]])
    
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
                continue 
            
            block_match = re.search(r"(ThermalUnitBlock|BatteryUnitBlock|IntermittentUnitBlock|HydroUnitBlock|DCNetworkBlock)\s*(\d*)", line)
            if block_match:
                base_block = block_match.group(1)
                block_number = block_match.group(2) or "0"  # Se non c'è numero, usa "0"

                block_key = f"{base_block}_{block_number}"  # Nome univoco del blocco

                if base_block != 'DCNetworkBlock':
                    if base_block not in data:
                        data[base_block] = {}  # Ora un dizionario invece di una lista
                    data[base_block][block_key] = {}  # Crea il nuovo blocco
                    current_block = block_key
                else:
                    current_block = 'DCNetworkBlock'
                continue  

            match = re.match(r"([\w\s]+?)(?:\s*\[(\d+)\])?\s+=\s+\[([^\]]*)\]", line)
            if match and current_block:
                key_base, sub_index, values = match.groups()
                key_base = key_base.strip()
            
                if current_block == 'DCNetworkBlock':
                    data[current_block]['PowerFlow'].extend([float(x) for x in values.split()])
                else:
                    base_block = current_block.split("_")[0]
                    block_data = data[base_block][current_block]
            
                    if sub_index is not None:
                        # Se la chiave esiste ed è già un array, va trasformata in un dizionario
                        if key_base in block_data and not isinstance(block_data[key_base], dict):
                            existing_value = block_data[key_base]
                            block_data[key_base] = {0: existing_value}  # Sposta il valore precedente come indice 0
                    
                        if key_base not in block_data:
                            block_data[key_base] = {}
                    
                        block_data[key_base][int(sub_index)] = np.array([float(x) for x in values.split()])
                    else:
                        # Se è già stato salvato come dizionario con indici, inseriamo in 0
                        if key_base in block_data and isinstance(block_data[key_base], dict):
                            block_data[key_base][0] = np.array([float(x) for x in values.split()])
                        else:
                            block_data[key_base] = np.array([float(x) for x in values.split()])

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
    
    network = one_bus_network(network)
    network.optimize(solver_name='gurobi')



