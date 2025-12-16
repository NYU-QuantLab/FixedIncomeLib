import os, csv
from abc import ABC
import datetime as dt
from typing import Self, Any
from fixedincomelib.date import Date
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

    def get_fixing(self, index : str, date : Date):
        this_map = self.get(index)
        if date in this_map:
            return this_map[date]
        else:
            raise Exception(f'Cannot find {index} for date ...') # ???

class IndexFixingsManager(Registry):

    _fixing_path = None

    def __new__(cls) -> Self:
        if cls._fixing_path is None:
            this_config = get_config()
            cls._fixing_path = this_config['FIXING_SOURCE']
        return super().__new__(cls, 'fixings', 'IndexFixings')
    
    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        this_path = os.path.join(self._fixing_path, f'{key}.csv')
        if not os.path.exists(this_path):
            with open(this_path, newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for this_line in csv_reader:
                    fixing_date = Date(dt.datetime.strptime(this_line['date'], '%m/%d/%Y').date())
                    self._map.setdefault(key.lower(), {})[fixing_date] = float(this_line["fixing"])

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
        super().register(key, value)
        type = value['type']
        value.pop('type')
        func = DataConventionRegFunction().get(type)
        self._map[key] = func(type, value)

############################################################################################



