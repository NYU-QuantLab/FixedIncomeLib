import os, csv, json
from abc import ABC
import datetime as dt
import pandas as pd
from typing import Self, Any, Optional
import QuantLib as ql
from fixedincomelib.date import Date, Period
from fixedincomelib.utilities import Registry, get_config

######################################### REGISTRY #########################################

class IndexRegistry(Registry):
    
    def __new__(cls) -> Self:
        return super().__new__(cls, 'indices', 'Index')

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        # delegate index to ql
        ql_object = None
        try:
            ql_object = getattr(ql, value)
        except AttributeError:
            raise KeyError(f"QuantLib has no attribute '{value}' for key '{key}'")
        self._map[key] = ql_object
        # self._reverse_map[ql_object.name()] = key

    def get(self, key: Any, **args) -> Any:
        try: 
            func = self._map[key.upper()]
            if 'term' in args:
                return func(Period(args['term']))
            else:
                return func()
        except:
            raise KeyError(f'no entry for key : {key}.')

    # def convert_ql_index(self, ql_index : str):
    #     if ql_index in self._reverse_map:
    #         return self._reverse_map[ql_index]
    #     raise Exception(f'Cannot map {ql_index} from QuantLib Index.')

    def display_all_indices(self) -> pd.DataFrame:
        default_term = '3M'
        to_print = []
        for k, _ in self._map.items():
            index = None
            try:
                index = self.get(k)
            except:
                index = self.get(k, term=default_term)
            index_name = index.name()
            if default_term in index_name:
                index_name = index_name.replace(default_term, '')
            to_print.append([k, index_name])
        return pd.DataFrame(to_print, columns=['Name', 'QuantLibIndex'])

class IndexFixingsManager(Registry):

    _fixing_path = None

    def __new__(cls) -> Self:
        if cls._fixing_path is None:
            this_config = get_config()
            cls._fixing_path = this_config['FIXING_SOURCE']
        return super().__new__(cls, 'fixings', 'IndexFixings')
    
    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        this_path = os.path.join(self._fixing_path, f'{key.lower()}.csv')
        if os.path.exists(this_path):
            with open(this_path, newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for this_line in csv_reader:
                    fixing_date = Date(dt.datetime.strptime(this_line['date'], '%m/%d/%Y').date())
                    self._map.setdefault(key.upper(), {})[fixing_date] = float(this_line["fixing"])
    
    def insert_fixing(self, index : str, date : Date, fixing : float):
        this_map = self.get(index.lower())
        if date in this_map:
            return
        else:
            this_map[date] = fixing

    def get_fixing(self, index : str, date : Date):
        this_map = self.get(index.lower())
        if date in this_map:
            return this_map[date]
        else:
            raise Exception(f'Cannot find {index} for date ...')
    
    def remove_fixing(self, index : str, date : Optional[Date]=None):
        if date is None:
            self.erase(index)
        else:
            this_map : dict = self.get(index)
            this_map.pop(Date(date))


class DataConventionRegFunction(Registry):

    def __new__(cls) -> Self:
        return super().__new__(cls, '', cls.__name__)

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        self._map[key] = value

class DataConventionRegistry(Registry):

    def __new__(cls) -> Self:
        return super().__new__(cls, 'data_conventions', 'DataConevention')
    
    def register(self, key : Any, value : Any) -> None:
        value_ = value.copy()
        super().register(key, value_)
        type = value_['type']
        value_.pop('type')
        func = DataConventionRegFunction().get(type)
        self._map[key] = func(key, value_)

    def display_all_data_conventions(self) -> pd.DataFrame:
        to_print = []
        for k, v in self._map.items():
            to_print.append([k, v.name])
        return pd.DataFrame(to_print, columns=['Name', 'Type'])

############################################################################################



