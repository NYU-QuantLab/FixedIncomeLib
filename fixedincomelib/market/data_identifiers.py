from typing import Tuple
from abc import ABC, abstractmethod
from fixedincomelib.market.data_conventions import *
from fixedincomelib.market.data_conventions import DataConvention


class DataIdentifier(ABC):

    _data_type = ''

    def __init__(self, data_convention : DataConvention) -> None:
        self.data_convention_ = data_convention
        self.data_identifier_ = (self._data_type, data_convention.name)
    
    @property
    def data_type(self) -> str:
        return self._data_type
    
    @property
    def data_convention(self) -> DataConvention:
        return self.data_convention_
    
    @property
    def data_identifier(self) -> Tuple[str, str]:
        return self.data_identifier_

    def to_string(self):
        return f'{self.data_type}:{self.data_convention.name}'
    
    @abstractmethod
    def unit(self):
        pass

class DataIdentifierOvernightIndexFuture(DataIdentifier):

    _data_type = 'Overnight Index Future'

    def __init__(self, data_convention: DataConventionRFRFuture) -> None:
        super().__init__(data_convention)

    def unit(self):
        return -0.01

class DataIdentifierOvernightIndexSwap(DataIdentifier):

    _data_type = 'Overnight Index Swap'

    def __init__(self, data_convention: DataConventionRFRSwap) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.0001

class DataIdentifierRFRJump(DataIdentifier):

    _data_type = 'Jump'

    def __init__(self, data_convention: DataConventionRFRJump) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.0001

    
class DataIdentifierSwaptionNormalVolatility(DataIdentifier):

    _data_type = 'Swaption Normal Volatility'

    def __init__(self, data_convention: DataConventionRFRSwaption) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.0001
    
class DataIdentifierSwaptionSABRBeta(DataIdentifier):

    _data_type = 'Swaption SABR Beta'

    def __init__(self, data_convention: DataConventionRFRSwaption) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.01
    
class DataIdentifierSwaptionSABRNu(DataIdentifier):

    _data_type = 'Swaption SABR Nu'

    def __init__(self, data_convention: DataConventionRFRSwaption) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.01
    
class DataIdentifierSwaptionSABRRho(DataIdentifier):

    _data_type = 'Swaption SABR Rho'

    def __init__(self, data_convention: DataConventionRFRSwaption) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.01

class DataIdentifierCapFloorNormalVolatility(DataIdentifier):

    _data_type = 'CapFloor Normal Volatility'

    def __init__(self, data_convention: DataConventionRFRCapFloor) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.0001
    
class DataIdentifierCapFloorSABRBeta(DataIdentifier):

    _data_type = 'CapFloor SABR Beta'

    def __init__(self, data_convention: DataConventionRFRCapFloor) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.01
    
class DataIdentifierCapFloorSABRNu(DataIdentifier):

    _data_type = 'CapFloor SABR Nu'

    def __init__(self, data_convention: DataConventionRFRCapFloor) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.01
    
class DataIdentifierCapFloorSABRRho(DataIdentifier):

    _data_type = 'CapFloor SABR Rho'

    def __init__(self, data_convention: DataConventionRFRCapFloor) -> None:
        super().__init__(data_convention)

    def unit(self):
        return 0.01


### registration
DataIdentifierRegistry().register(DataIdentifierOvernightIndexFuture._data_type.upper(), DataIdentifierOvernightIndexFuture)
DataIdentifierRegistry().register(DataIdentifierOvernightIndexSwap._data_type.upper(), DataIdentifierOvernightIndexSwap)
DataIdentifierRegistry().register(DataIdentifierRFRJump._data_type.upper(), DataIdentifierRFRJump)
DataIdentifierRegistry().register(DataIdentifierSwaptionNormalVolatility._data_type.upper(), DataIdentifierSwaptionNormalVolatility)
DataIdentifierRegistry().register(DataIdentifierSwaptionSABRBeta._data_type.upper(), DataIdentifierSwaptionSABRBeta)
DataIdentifierRegistry().register(DataIdentifierSwaptionSABRNu._data_type.upper(), DataIdentifierSwaptionSABRNu)
DataIdentifierRegistry().register(DataIdentifierSwaptionSABRRho._data_type.upper(), DataIdentifierSwaptionSABRRho)
DataIdentifierRegistry().register(DataIdentifierCapFloorNormalVolatility._data_type.upper(), DataIdentifierCapFloorNormalVolatility)
DataIdentifierRegistry().register(DataIdentifierCapFloorSABRBeta._data_type.upper(), DataIdentifierCapFloorSABRBeta)
DataIdentifierRegistry().register(DataIdentifierCapFloorSABRNu._data_type.upper(), DataIdentifierCapFloorSABRNu)
DataIdentifierRegistry().register(DataIdentifierCapFloorSABRRho._data_type.upper(), DataIdentifierCapFloorSABRRho)