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


def parse_txt_file(file_path):
    data = {'DCNetworkBlock': {'PowerFlow': []}}  # Inizializza con una lista temporanea
    current_block = None  

    with open(file_path, "r") as file:
        for line in file:
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



