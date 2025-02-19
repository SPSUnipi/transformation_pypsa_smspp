# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:38 2024

@author: aless
"""

import sys
import os

# Aggiunge il percorso relativo per la cartella `config`
sys.path.append(os.path.abspath("../configs"))
# Aggiunge il percorso relativo per la cartella `scripts`
sys.path.append(os.path.abspath("."))


from config import Config
from network_definition import NetworkDefinition
from transformation import Transformation
from datetime import datetime
from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig

#%% Network definition with PyPSA
config = Config()
nd = NetworkDefinition(config)

network = nd.n.copy()
network.optimize(solver_name='gurobi')

#%% Transformation class
then = datetime.now()
transformation = Transformation(network)
print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")

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
line_variables = {}
for name, variable in transformation.networkblock['Lines'][1].items():
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
i = 0
for name, unit_block in transformation.unitblocks.items():
    kwargs = {}
    for variable_name, variable in unit_block[1].items():
        kwargs[variable_name] = Variable(
            variable_name,
            variable['type'],
            variable['size'],
            variable['value'])
    
    unit_block = Block().from_kwargs(
        block_type=unit_block[0],
        **kwargs
    )
    
    # Why should I have name UnitBlock_0?
    sn.blocks["Block_0"].add_block(f"UnitBlock_{i}", block=unit_block)
    i += 1
    
    
# Optimization
configfile = SMSConfig(
    template="uc_solverconfig"
)  # path to the template solver config file "uc_solverconfig"
temporary_smspp_file = "./2buses_2thermal.nc"  # path to temporary SMS++ file
output_file = "./2buses_2thermal.txt"  # path to the output file (optional)

result = sn.optimize(
    configfile,
    temporary_smspp_file,
    output_file,
)

