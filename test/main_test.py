# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:38 2024

@author: aless
"""

import sys
import os

# Aggiunge il percorso relativo per la cartella `config`
sys.path.append(os.path.abspath("../scripts"))
# Aggiunge il percorso relativo per la cartella `scripts`
sys.path.append(os.path.abspath("."))

from pypsa2smspp import Transformation
from datetime import datetime
from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig
import pypsa
from pypsa2smspp import (
    clean_marginal_cost,
    clean_global_constraints,
    clean_e_sum,
    clean_efficiency_link,
    clean_ciclicity_storage,
    clean_marginal_cost_intermittent,
    clean_storage_units,
    clean_stores,
    clean_p_min_pu,
    one_bus_network,
    parse_txt_file
    )

#%% Network definition with PyPSA
network_name = "microgrid_microgrid_ALL_1N"
network = pypsa.Network(f"test/networks/{network_name}.nc")

network = clean_marginal_cost(network)
network = clean_global_constraints(network)
network = clean_e_sum(network)
network = clean_ciclicity_storage(network)

# network = one_bus_network(network)
# network = clean_efficiency_link(network)


# network = clean_p_min_pu(network)
# network = clean_marginal_cost_intermittent(network)
# network = clean_storage_units(network)
# network = clean_stores(network)


network.optimize(solver_name='highs')
# network.export_to_netcdf()


#%% Transformation class
then = datetime.now()
transformation = Transformation(network)
print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")

# %% SMSpp optimization

# pySMSpp
sn = SMSNetwork(file_type=SMSFileType.eBlockFile) # Empty Block

# Dimensions of the problem
kwargs = transformation.dimensions

# Load
demand_name = transformation.demand['name']
demand_type = transformation.demand['type']
demand_size = transformation.demand['size']
demand_value = transformation.demand['value']

demand = {demand_name: Variable(  # active power demand
        demand_name,
        demand_type,
        demand_size,
        demand_value )}

kwargs = {**kwargs, **demand}

# Generator node
generator_node = {transformation.generator_node['name']: Variable(
    transformation.generator_node['name'],
    transformation.generator_node['type'],
    transformation.generator_node['size'],
    transformation.generator_node['value'])}

kwargs = {**kwargs, **generator_node}

# Lines
if kwargs['NumberLines'] > 0:
    line_variables = {}
    for name, variable in transformation.networkblock['Lines']['variables'].items():
        line_variables[name] = Variable(
            name,
            variable['type'],
            variable['size'],
            variable['value'])
        
    kwargs = {**kwargs, **line_variables}

# Add UC block
sn.add(
    "UCBlock",  # block type
    "Block_0",  # block name
    id="0",  # block id
    **kwargs
)

# Add unit blocks

for name, unit_block in transformation.unitblocks.items():
    kwargs = {}
    for variable_name, variable in unit_block['variables'].items():
        kwargs[variable_name] = Variable(
            variable_name,
            variable['type'],
            variable['size'],
            variable['value'])
        
    if 'dimensions' in unit_block.keys():
        for dimension_name, dimension in unit_block['dimensions'].items():
            kwargs[dimension_name] = dimension
    
    unit_block_toadd = Block().from_kwargs(
        block_type=unit_block['block'],
        **kwargs
    )
    
    # Why should I have name UnitBlock_0?
    sn.blocks["Block_0"].add_block(unit_block['enumerate'], block=unit_block_toadd)
    
    
# Optimization
configfile = SMSConfig(
    template="uc_solverconfig"
)  # path to the template solver config file "uc_solverconfig"
temporary_smspp_file = f"./output_files/{network_name}.nc"  # path to temporary SMS++ file
output_file = f"./output_files/{network_name}.txt"  # path to the output file (optional)

import os
os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

result = sn.optimize(
    configfile,
    temporary_smspp_file,
    output_file,
)

# Esegui la funzione sul file di testo
data_dict = parse_txt_file(output_file)

print(f"Il solver ci ha messo {data_dict['elapsed_time']}s")
print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")

statistics = network.statistics()
operational_cost = statistics['Operational Expenditure'].sum()
error = (operational_cost - result.objective_value) / operational_cost * 100
print(f"Error PyPSA-SMS++ of {error}%")