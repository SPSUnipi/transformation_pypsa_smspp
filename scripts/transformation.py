# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import pypsa
import numpy as np
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
import re

NP_DOUBLE = np.float64
NP_UINT = np.uint32

class Transformation:
    """
    Transformation class for converting the components of a PyPSA energy network into unit blocks.
    In particular, these are ready to be implemented in SMS++

    The class takes as input a PyPSA network.
    It reads the specified network components and converts them into a dictionary of unit blocks (`unitblocks`).
    
    Attributes:
    ----------
    unitblocks : dict
        Dictionary that holds the parameters for each unit block, organized by network components.
    
    IntermittentUnitBlock_parameters : dict
        Parameters for an IntermittentUnitBlock, like solar and wind turbines.
        The values set to a float number are absent in Pypsa, while lambda functions are used to get data from
        Pypa DataFrames
    
    ThermalUnitBlock_parameters : dict
        Parameters for a ThermalUnitBlock
    """

    def __init__(self, n, max_hours_stores=1):
        """
        Initializes the Transformation class.

        Parameters:
        ----------
        
        n : PyPSA Network
            PyPSA energy network object containing components such as generators and storage units.
        max_hours_stores: int
            Max hours parameter for stores, default is 10h. Stores do not have this parameter, but it is required to model them as BatteryUnitBlocks
        
        Methods:
        ----------
        init : Start the workflow of the class
        
        """
        
        # Attribute for unit blocks
        self.unitblocks = dict()
        self.networkblock = dict()
        
        # Parameters for intermittent units
        self.IntermittentUnitBlock_parameters = {
            "Gamma": 0.0,
            "Kappa": 1.0,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "InertiaPower": 1.0
        }
        
        # Parameters for thermal units
        self.ThermalUnitBlock_parameters = {
            "InitUpDownTime": lambda up_time_before: up_time_before,
            "MinUpTime": lambda min_up_time: min_up_time,
            "MinDownTime": lambda min_down_time: min_down_time, 
            #"DeltaRampUp": lambda ramp_limit_up: ramp_limit_up if not np.isnan(ramp_limit_up) else 0,
            #"DeltaRampDown": lambda ramp_limit_down: ramp_limit_down if not np.isnan(ramp_limit_down) else 0,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "PrimaryRho": 0.0,
            "SecondaryRho": 0.0,
            "Availability": 1,
            "QuadTerm": lambda marginal_cost_quadratic: marginal_cost_quadratic,
            "LinearTerm": lambda marginal_cost: marginal_cost,
            "ConstantTerm": 0.0,
            "StartUpCost": lambda start_up_cost: start_up_cost,
            "InitialPower": lambda p: p[0][0],
            "FixedConsumption": 0.0,
            "InertiaCommitment": 1.0
        }
        
        self.BatteryUnitBlock_parameters = {
            "Kappa": 1.0,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "ExtractingBatteryRho": lambda efficiency_dispatch: 1 / efficiency_dispatch,
            "StoringBatteryRho": lambda efficiency_store: efficiency_store,
            "Demand": 0.0,
            "MinStorage": 0.0,
            "MaxStorage": lambda p_nom_opt, p_max_pu, max_hours: p_nom_opt * p_max_pu * max_hours,
            "MaxPrimaryPower": 0.0,
            "MaxSecondaryPower": 0.0,
            "InitialPower": lambda p: p[0][0],
            "InitialStorage": lambda state_of_charge, cyclic_state_of_charge: -1 if cyclic_state_of_charge.values else state_of_charge[0][0],
            # "Cost": lambda marginal_cost: marginal_cost
            }
        
        self.BatteryUnitBlock_store_parameters = {
            "Kappa": 1.0,
            "MaxPower": lambda e_nom_opt, e_max_pu, max_hours: e_nom_opt * e_max_pu / max_hours,
            "MinPower": lambda e_nom_opt, e_max_pu, max_hours: - e_nom_opt * e_max_pu / max_hours,
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "ExtractingBatteryRho": lambda e_max_pu: np.ones_like(e_max_pu),
            "StoringBatteryRho": lambda e_max_pu: np.ones_like(e_max_pu),
            "Demand": 0.0,
            "MinStorage": 0.0,
            "MaxStorage": lambda e_nom_opt: e_nom_opt,
            "MaxPrimaryPower": 0.0,
            "MaxSecondaryPower": 0.0,
            "InitialPower": lambda e_initial, max_hours: e_initial / max_hours,
            "InitialStorage": lambda e_initial, e_cyclic: -1 if e_cyclic.values else e_initial,
            #"Cost": lambda capital_cost: capital_cost
            }
                
        self.Lines_parameters = {
            "StartLine": lambda start_line_idx: start_line_idx.values,
            "EndLine": lambda end_line_idx: end_line_idx.values,
            "MinPowerFlow": lambda s_nom_opt: -s_nom_opt.values,
            "MaxPowerFlow": lambda s_nom_opt: s_nom_opt.values,
            "LineSusceptance": lambda s_nom_opt: np.zeros_like(s_nom_opt)
            }
        
        self.Links_parameters = {
            "StartLine": lambda start_line_idx: start_line_idx.values,
            "EndLine": lambda end_line_idx: end_line_idx.values,
            "MinPowerFlow": lambda p_nom_opt, p_min_pu: p_nom_opt.values * p_min_pu.values,
            "MaxPowerFlow": lambda p_nom_opt, p_max_pu: p_nom_opt.values * p_max_pu.values,
            "LineSusceptance": lambda s_nom_opt: np.zeros_like(s_nom_opt)
            }

        self.HydroUnitBlock_parameters = {
            "StartArc": lambda p_nom: np.full(len(p_nom)*3, 0),
            "EndArc": lambda p_nom: np.full(len(p_nom)*3, 1),
            "MaxVolumetric": lambda p_nom_opt, max_hours: (p_nom_opt * max_hours),
            "MinVolumetric": 0.0,
            "Inflows": lambda inflow: inflow.values.transpose(),
            "MaxFlow": lambda p_nom_opt, p_max_pu, inflow: (np.array([100* (p_nom_opt*p_max_pu), 100*np.full_like(p_max_pu, inflow.max()), (0.*p_max_pu)])).squeeze().transpose(),
            "MinFlow": lambda inflow: np.array([0., 0., -100*inflow.values.max()]),
            "MaxPower": lambda p_nom_opt, p_max_pu: (np.array([(p_nom_opt*p_max_pu), (0.*p_max_pu), (0. *p_max_pu)])).squeeze().transpose(),
            "MinPower": lambda p_nom_opt, p_min_pu: (np.array([(p_nom_opt*p_min_pu), (0.*p_min_pu), (0. *p_min_pu)])).squeeze().transpose(),
            # "PrimaryRho": lambda p_nom: np.full(len(p_nom)*3, 0.),
            # "SecondaryRho": lambda p_nom: np.full(len(p_nom)*3, 0.),
            "NumberPieces": lambda p_nom: np.full(len(p_nom)*3, 1),
            "ConstantTerm": lambda p_nom: np.full(len(p_nom)*3, 0),
            "LinearTerm": lambda efficiency_dispatch, efficiency_store: np.array([1/efficiency_dispatch.values.max(), 0., efficiency_store.values.max()]),
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "DownhillFlow": lambda p_nom: np.full(len(p_nom)*3, 0.),
            "UphillFlow": lambda p_nom: np.full(len(p_nom)*3, 0.),
            #"InertiaPower": 1.0,
            # "InitialFlowRate": lambda inflow: inflow.values[0],
            "InitialVolumetric": lambda state_of_charge_initial, max_hours, p_nom_opt: state_of_charge_initial * max_hours * p_nom_opt
            }
        
        # If the PyPSA DataFrame does not contain a value, it is taken from this dict
        # It collects defaults from https://pypsa.readthedocs.io/en/latest/user-guide/components.html
        # self.default_values = {
        #     "up_time_before": 1.0,
        #     "min_up_time": 0.0,
        #     "min_down_time": 0.0,
        #     "ramp_limit_up": np.nan,
        #     "ramp_limit_down": np.nan,
        #     "marginal_cost_quadratic": 0.0,
        #     "marginal_cost": 0.0,
        #     "start_up_cost": 0.0,
        #     "efficiency_dispatch": 1.0,
        #     "efficiency_store": 1.0,
        #     "capital_cost": 0.0,
        #     "p_nom_opt": 0,
        #     }
        
        self.conversion_dict = {
            "T": "TimeHorizon",
            "NU": "NumberUnits",
            "NE": "NumberElectricalGenerators",
            "N": "NumberNodes",
            "L": "NumberLines",
            "Li": "NumberLinks",
            "NA": "NumberArcs",
            "NR": "NumberReservoirs",
            "NP": "TotalNumberPieces",
            "1": 1
            }
        
        n.stores["max_hours"] = max_hours_stores
        
        # Initialize with the parser and network
        self.init(n)
        
    def init(self, n):
        """
        Initialization method describing the workflow of the class.

        Parameters:
        ----------

        n : PyPSA Network
            PyPSA network object for the energy network.
            
        Methods : iterate_components
            It iterates over all the components to convert them into UnitBlocks
        """
        
        self.remove_zero_p_nom_opt_components(n)
        self.read_excel_components()
        self.add_dimensions(n)
        self.iterate_components(n)
        self.add_demand(n)
        self.lines_links()

    def get_paramer_as_dense(self, n, component, field, weights=True):
        """
        Get the parameters of a component as a dense DataFrame
    
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network
        component : str
            The component to get the parameters from
        field : str
            The field to get the parameters from
    
        Returns
        -------
        pd.DataFrame
            The parameters of the component as a dense DataFrame
        """
        
        sns = n.snapshots
        
        # Related to different investment periods
        if not n.investment_period_weightings.empty:  # TODO: check with different version
            periods = sns.unique("period")
            period_weighting = n.investment_period_weightings.objective[periods]
        weighting = n.snapshot_weightings.objective
        if not n.investment_period_weightings.empty:
            weighting = weighting.mul(period_weighting, level=0).loc[sns]
        else:
            weighting = weighting.loc[sns]
         
        # If static, it will be expanded
        if field in n.static(component).columns:
            field_val = get_as_dense(n, component, field, sns)
        else:
            field_val = n.dynamic(component)[field]
        
        # If economic, it will be weighted
        if weights:
            field_val = field_val.mul(weighting, axis=0)
        return field_val 
    
    
    def add_demand(self, n):
        demand = n.loads_t.p_set.rename(columns=n.loads.bus)
        demand = demand.T.reindex(n.buses.index).fillna(0.)
        self.demand = {'name': 'ActivePowerDemand', 'type': 'float', 'size': ("NumberNodes", "TimeHorizon"), 'value': demand}
        
    def add_dimensions(self, n):
        components = {
            "NumberUnits": ["generators", "storage_units", "stores"],
            "NumberElectricalGenerators": ["generators", "storage_units", "stores"],
            "NumberNodes": ["buses"],
            "NumberLines": ["lines", "links"],
        }
        
        # Calcola le dimensioni sommando le lunghezze
        self.dimensions = {
            "TimeHorizon": len(n.snapshots),
            **{
                name: sum(len(getattr(n, comp)) for comp in comps)
                for name, comps in components.items()
            },
            
            }
        
        
        components = {
            "Lines": ['lines'],
            "Links": ['links'],
            "combined": ['lines', 'links']
            }
        self.dimensions_lines = {
            **{
                name: sum(len(getattr(n, comp)) for comp in comps)
                for name, comps in components.items()
            }
            }
        
        
    def add_UnitBlock(self, attr_name, components_df, components_t, components_type, n, component=None, index=None):
        """
        Adds a unit block to the `unitblocks` dictionary for a given component.

        Parameters:
        ----------
        attr_name : str
            Attribute name containing the unit block parameters (Intermittent or Thermal).
        
        components_df : DataFrame
            DataFrame containing information for a single component.
            For example, n.generators.loc['wind']

        components_t : DataFrame
            Temporal DataFrame (e.g., snapshot) for the component.
            For example, n.generators_t

        Sets:
        --------
        self.unitblocks[components_df.name] : dict
            Dictionary of transformed parameters for the component.
        """
        converted_dict = {}
        if hasattr(self, attr_name):
            unitblock_parameters = getattr(self, attr_name)
        else:
            print("Block not yet implemented") # TODO: Replace with logger
            
        
        for key, func in unitblock_parameters.items():
            if callable(func):
                # Extract parameter names from the function
                param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                args = []
                
                for param in param_names:
                    if self.smspp_parameters[attr_name.split("_")[0]]['Size'][key] not in [1, '[L]', '[Li]', '[NA]', '[NP]']:
                        weight = True if param in ['capital_cost', 'marginal_cost', 'marginal_cost_quadratic', 'start_up_cost', 'stand_by_cost'] else False
                        arg = self.get_paramer_as_dense(n, components_type, param, weight)[[component]]
                    elif param in components_df.index or param in components_df.columns:
                        arg = components_df.get(param)
                    elif param in components_t.keys():
                        df = components_t[param]
                        arg = df[components_df.index].values
                    args.append(arg)
                
                # Apply function to the parameters
                value = func(*args)
                value = value[components_df.index].values if isinstance(value, pd.DataFrame) else value
                value = value.item() if isinstance(value, pd.Series) else value
                variable_type, variable_size = self.add_size_type(attr_name, key, value)
                converted_dict[key] = {"value": value, "type": variable_type, "size": variable_size}
            else:
                value = func
                variable_type, variable_size = self.add_size_type(attr_name, key)
                converted_dict[key] = {"value": value, "type": variable_type, "size": variable_size}
        
        name = components_df.name if isinstance(components_df, pd.Series) else attr_name.split("_")[0]
        
        if attr_name in ['Lines_parameters', 'Links_parameters']:
            self.networkblock[name] = {"block": 'Lines', "variables": converted_dict}
        else:
            self.unitblocks[components_df.index[0]] = {"enumerate": f"UnitBlock_{index}" ,"block": attr_name.split("_")[0], "variables": converted_dict}   
        
        if attr_name == 'HydroUnitBlock_parameters':
            dimensions = self.hydro_dimensions()
            self.unitblocks[components_df.index[0]]['dimensions'] = dimensions
            
    
    def hydro_dimensions(self):
        dimensions = {}
        dimensions["NumberReservoirs"] = 1
        dimensions["NumberArcs"] = 3 * dimensions["NumberReservoirs"]
        dimensions["TotalNumberPieces"] = 3
        self.dimensions["NumberElectricalGenerators"] += 2*dimensions["NumberReservoirs"] 
        
        return dimensions
            
    
    def remove_zero_p_nom_opt_components(self, n):
        # Lista dei componenti che hanno l'attributo p_nom_opt
        components_with_p_nom_opt = ["Generator", "Link", "Store", "StorageUnit", "Line", "Transformer"]
        nominal_attrs = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        
        for components in n.iterate_components(["Line", "Generator", "Link", "Store", "StorageUnit"]):
            components_df = components.df
            components_df = components_df[components_df[f"{nominal_attrs[components.name]}_opt"] > 0]
            setattr(n, components.list_name, components_df)

    
    def iterate_components(self, n):
        """
        Iterates over the network components and adds them as unit blocks.

        Parameters:
        ----------
        n : PyPSA Network
            PyPSA network object containing components to iterate over.
            
        Methods: add_UnitBlock
            Method to convert the DataFrame and get a UnitBlock
        
        Adds:
        ---------
        The components to the `unitblocks` dictionary, with distinct attributes for intermittent and thermal units.
        
        """
        renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'PV', 'wind', 'ror']
        nominal_attrs = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        
        generator_node = []
        index = 0
        for components in n.iterate_components(["Line", "Generator", "Link", "Store", "StorageUnit"]):
            # Static attributes of the class of components
            components_df = components.df
            # Dynamic attributes of the class of components
            components_t = components.dynamic
            # Class of components
            components_type = components.list_name
            # Get the index for each component (useful especially for lines)
            if components_type == 'lines':
                self.get_bus_idx(n, components_df, components_df.bus0, "start_line_idx")
                self.get_bus_idx(n, components_df, components_df.bus1, "end_line_idx")
                attr_name = "Lines_parameters"
                self.add_UnitBlock(attr_name, components_df, components_t, components.name, n)
                continue
            elif components_type == 'links':
                self.get_bus_idx(n, components_df, components_df.bus0, "start_line_idx")
                self.get_bus_idx(n, components_df, components_df.bus1, "end_line_idx")
                attr_name = "Links_parameters"
                self.add_UnitBlock(attr_name, components_df, components_t, components.name, n)
                continue
            elif components_type == 'storage_units':
                self.get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                for bus, carrier in zip(components_df['bus_idx'].values, components_df['carrier']):
                    if carrier == 'hydro':
                        generator_node.extend([bus] * 3)  # Repeat three times
                    else:
                        generator_node.append(bus)  # Normal case
            else:
                self.get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                generator_node.extend(components_df['bus_idx'].values)


                # Understand which type of block we expect

            for component in components_df.index:
                if any(carrier in components_df.loc[component].carrier for carrier in renewable_carriers):
                    attr_name = "IntermittentUnitBlock_parameters"
                elif components_df.loc[component].carrier == 'hydro':
                    attr_name = 'HydroUnitBlock_parameters'
                elif "storage_units" in components_type:
                    attr_name = "BatteryUnitBlock_parameters"
                elif "store" in components_type:
                    attr_name = "BatteryUnitBlock_store_parameters"
                else:
                    attr_name = "ThermalUnitBlock_parameters"
                
                self.add_UnitBlock(attr_name, components_df.loc[[component]], components_t, components.name, n, component, index)
                index += 1
        self.generator_node = {'name': 'GeneratorNode', 'type': 'float', 'size': ("NumberElectricalGenerators",), 'value': generator_node}
        
    def get_bus_idx(self, n, components_df, bus_series, column_name, dtype="uint32"):
        """
        Returns the numeric index of the bus in the network n for each element of the bus_series.
        ----------
        n : PyPSA Network
        bus_series : series of buses. For example, n.lines.bus0 o n.generators.bus
        ----------
        Example: one single bus with two generators (wind and diesel 1)
                    n.generators.bus.map(n.buses.index.get_loc).astype("uint32")
                    Generator
                    wind        0
                    diesel 1    0
                    Name: bus, dtype: uint32
        """
        components_df[column_name] = bus_series.map(n.buses.index.get_loc).astype(dtype).values

    def read_excel_components(self):
        """
        Reads Excel file for size and type of SMS++ parameters. Each sheet includes a class of components

        Returns:
        ----------
        all_sheets : dict
            Dictionary where keys are sheet names and values are DataFrames containing 
            data for each UnitBlock type (or lines).
        """
        file_path = "../data/parameters_smspp.xlsx"
        self.smspp_parameters = pd.read_excel(file_path, sheet_name=None, index_col=0)
        
    
    def add_size_type(self, attr_name, key, args=None):
        """
        Adds the size and dtype of a variable (for the NetCDF file) based on the Excel file information.
        """
        # Ottieni i parametri del tipo di blocco e la riga corrispondente
        row = self.smspp_parameters[attr_name.split("_")[0]].loc[key]
        variable_type = row['Type']
        
        dimensions = self.dimensions.copy()
        
        # Useful only for this case. If variable, a solution must be found
        dimensions[1] = 1
        dimensions["NumberReservoirs"] = 1
        dimensions["NumberArcs"] = 3 * dimensions["NumberReservoirs"]
        dimensions["TotalNumberPieces"] = 3
        dimensions["NumberLines"] = self.dimensions_lines['Lines']
        dimensions["NumberLinks"] = self.dimensions_lines['Links']
    
        # Determina la dimensione della variabile
        if args is None:
            variable_size = ()
        else:
            # Se args è un numero scalare, la dimensione è 1
            if isinstance(args, (float, int, np.integer)):
                variable_size = ()
            else:
                # Ottieni la forma se args è un array numpy
                if isinstance(args, np.ndarray):
                    shape = args.shape
                else:
                    shape = (len(args),)  # Se è una lista, trattala come un vettore
    
                # Estrai le dimensioni attese dal file Excel
                size_arr = re.sub(r'\[|\]', '', str(row['Size']).replace("][", ","))
                size_arr = size_arr.replace(" ", "").split("|")
    
                for size in size_arr:
                    if size == '1' and shape == (1,):
                        variable_size = ()
                        break
                    else:
                        # Scomponi espressioni tipo "T,L"
                        size_components = size.split(",")
                        expected_shape = tuple(dimensions[self.conversion_dict[s]] for s in size_components if s in self.conversion_dict)
    
                        if shape == expected_shape:
                            if "1" in size_components or len(size_components) == 1:
                                variable_size = (self.conversion_dict[size_components[0]],)  # Vettore
                            else:
                                variable_size = (self.conversion_dict[size_components[0]], self.conversion_dict[size_components[1]])  # Matrice
                            break
        return variable_type, variable_size
        
    def lines_links(self):
        if "Lines" in self.networkblock and "Links" in self.networkblock:
            for key, value in self.networkblock['Lines']['variables'].items():
                # Required to avoid problems for line susceptance
                if not isinstance(self.networkblock['Lines']['variables'][key]['value'], (int, float, np.integer)):
                    self.networkblock['Lines']['variables'][key]['value'] = np.concatenate([
                        self.networkblock["Lines"]['variables'][key]['value'], 
                        self.networkblock["Links"]['variables'][key]['value']
                    ])
            self.networkblock.pop("Links", None)
    
        elif "Links" in self.networkblock and "Lines" not in self.networkblock:
            # Se ci sono solo i Links, rinominali in Lines
            self.networkblock["Lines"] = self.networkblock.pop("Links")
            for key, value in self.networkblock['Lines']['variables'].items():
                value['size'] = tuple('NumberLines' if x == 'NumberLinks' else x for x in value['size'])
            
            
            

        