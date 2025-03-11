# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 08:23:19 2025

@author: aless
"""

import sys
import os

# Aggiunge il percorso relativo per la cartella `config`
sys.path.append(os.path.abspath("../scripts"))
# Aggiunge il percorso relativo per la cartella `scripts`
sys.path.append(os.path.abspath("."))

from transformation import Transformation
from datetime import datetime
from pysmspp import Variable, Block
import pypsa

def get_network(network_name):
    ''' Upload the PyPSA network '''
    network = pypsa.Network(f"networks/{network_name}.nc")
    return network 

def optimize_network(network, solver_name='gurobi'):
    ''' Optimize the PyPSA network '''
    network.optimize(solver_name=solver_name)
    
def transformation_class(network):
    ''' Convert the PyPSA network with the transformation class '''
    then = datetime.now()
    transformation = Transformation(network)
    print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")
    return transformation
    
def get_transformation_demand(transformation):
    ''' Get the demand from the transformation class '''
    demand = {transformation.demand['name']: Variable(  # active power demand
            transformation.demand['name'],
            transformation.demand['type'],
            transformation.demand['size'],
            transformation.demand['value'])}
    return demand

def get_transformation_dimensions(transformation):
    ''' Get the dimensions from the transformation class
        TimeHorizon, NumberUnits, NumberElectricalGenerators, NumberNodes, NumberLines
    '''
    return transformation.dimensions

def get_lines_variables(transformation):
    ''' Get the variables for the lines, inserted in the UCBlock '''
    line_variables = {}
    for name, variable in transformation.networkblock['Lines']['variables'].items():
        line_variables[name] = Variable(
            name,
            variable['type'],
            variable['size'],
            variable['value'])
    return line_variables

def get_generator_node(transformation):
    ''' Get the variable generator node from the transformation class '''
    generator_node = {transformation.generator_node['name']: Variable(
        transformation.generator_node['name'],
        transformation.generator_node['type'],
        transformation.generator_node['size'],
        transformation.generator_node['value'])}
    return generator_node

def add_ucblock(sn, kwargs):
    ''' Add UC block '''
    return sn.add(
        "UCBlock",  # block type
        "Block_0",  # block name
        id="0",  # block id
        **kwargs
    )

def build_unitblock(unit_block):
    ''' Build a unitblock (hydro, intermittent, thermal, battery) '''
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
    
    return Block().from_kwargs(
        block_type=unit_block['block'],
        **kwargs
    )

def add_unitblock_toucblock(sn, unit_block):
    ''' Add a unitblock to a ucblock '''
    unit_block_toadd = build_unitblock(unit_block)
    sn.blocks["Block_0"].add_block(unit_block['enumerate'], block=unit_block_toadd)
    return sn

def comparison_pypsa_smspp(network, result):
    ''' Compares the results for PyPSA and SMSpp networks '''
    statistics = network.statistics()
    operational_cost = statistics['Operational Expenditure'].sum()
    error = (operational_cost - result.objective_value) / operational_cost * 100
    print(f"Error PyPSA-SMS++ of {error}%")