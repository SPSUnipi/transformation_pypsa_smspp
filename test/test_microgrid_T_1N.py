# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:24:16 2025

@author: aless
"""

from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig
from conftest import (
    get_network,
    optimize_network,
    transformation_class,
    get_transformation_demand,
    get_transformation_dimensions,
    get_lines_variables,
    get_generator_node,
    add_ucblock,
    add_unitblock_toucblock,
    comparison_pypsa_smspp
    )

import pypsa
import pytest


def optimize_pypsa_smspp_network(network_name='microgrid_microgrid_T_1N'):
    
    network = get_network(network_name)
    optimize_network(network)
    
    transformation = transformation_class(network)
    
    # pySMSpp
    sn = SMSNetwork(file_type=SMSFileType.eBlockFile) # Empty Block
    
    # Retrieve transformation information
    dimensions = get_transformation_dimensions(transformation)
    demand = get_transformation_demand(transformation)
    generator_node = get_generator_node(transformation)
    
    kwargs = {**dimensions, **demand, **generator_node}
    
    
    if dimensions['NumberLines'] > 0:
        line_variables = get_lines_variables(transformation)
        kwargs = {**kwargs, **line_variables}
        
    sn = add_ucblock(sn, kwargs)
    
    for name, unit_block in transformation.unitblocks.items():
        add_unitblock_toucblock(sn, unit_block)
        
    
    configfile = SMSConfig(
        template="uc_solverconfig"
    )  # path to the template solver config file "uc_solverconfig"
    temporary_smspp_file = f"./output_files/{network_name}.nc"  # path to temporary SMS++ file
    output_file = f"./output_files/{network_name}.txt"  # path to the output file (optional)
 
    result = sn.optimize(
        configfile,
        temporary_smspp_file,
        output_file,
    )
        
    comparison_pypsa_smspp(network, result)
    
    
    assert "success" in result.status.lower()
    assert "error" not in result.log.lower()
    assert "ThermalUnitBlock" in result.log
    assert "BatteryUnitBlock" not in result.log
    assert "HydroUnitBlock" not in result.log
    assert "IntermittentUnitBlock" not in result.log
