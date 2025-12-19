from typing import Union, Dict, Any
from datetime import datetime
from fixedincomelib.date import TermOrTerminationDate, Date
from fixedincomelib.market import *
from fixedincomelib.product.linear_products import ProductRfrFuture, ProductRfrSwap
from fixedincomelib.product.product_display_visitor import ProductDisplayVisitor


class ProductFactory:
    _BUILDER_REGISTRY: Dict[str, str] = {
        "RFR FUTURE": "createRFRFuture",
        "RFR SWAP": "createRFRSwap",
    }

    @classmethod
    def registerBuilder(cls, data_type: str, builder_method_name: str):
        cls._BUILDER_REGISTRY[data_type] = builder_method_name

    @classmethod
    def getSupportedTypes(cls) -> list:
        return list(cls._BUILDER_REGISTRY.keys())


    @classmethod
    def createProductFromDataConvention(
        cls,
        axis1: str,
        dataConvention: Union[str, DataConvention],
        values: float,
        **kwargs: Any,
    ):
        if isinstance(dataConvention, str):
            convention_obj = DataConventionRegistry().get(dataConvention)
            if convention_obj is None:
                raise ValueError(f"Data convention '{dataConvention}' cannot be found in registry.")
        else:
            convention_obj = dataConvention

        data_type = convention_obj.data_type

        builder_method_name = cls._BUILDER_REGISTRY.get(data_type)
        if builder_method_name is None:
            raise ValueError(
                f"Data convention type: '{data_type}' not in _BUILDER_REGISTRY. "
                f"_BUILDER_REGISTRY includes: {list(cls._BUILDER_REGISTRY.keys())}"
            )

        builder_method = getattr(cls, builder_method_name)
        return builder_method(axis1, convention_obj, values, **kwargs)


    @classmethod
    def createRFRFuture(
        cls,
        axis1: str,
        dataConvention: DataConvention,
        values: float,
        **kwargs: Any,
    ) -> ProductRfrFuture:
        effectiveDate, termOrEnd = cls._parseAxis1(axis1)

        index = dataConvention.index
        compounding = cls._inferCompoundingMethod(dataConvention)

        longOrShort = kwargs.get("longOrShort")
        if longOrShort is None:
            raise ValueError("longOrShort parameter is missing for RFR Future")

        price = float(values)
        strike = kwargs.get("strike", price)

        contractualSize = dataConvention.contractual_notional
        notional = kwargs.get("notional", contractualSize)

        accrued_flag = kwargs.get("accrued_flag", -1.0)

        return ProductRfrFuture(
            effectiveDate=effectiveDate,
            termOrEnd=termOrEnd,
            index=index,
            compounding=compounding,
            longOrShort=longOrShort,
            strike=strike,
            notional=notional,
            contractualSize=contractualSize,
            accrued_flag=accrued_flag,
        )


    @classmethod
    def createRFRSwap(
        cls,
        axis1: str,
        dataConvention: DataConvention,
        values: float,
        **kwargs: Any,
    ) -> ProductRfrSwap:
        effectiveDate, termOrEnd = cls._parseAxis1(axis1)

        if effectiveDate is None:
            effectiveDate = Date.todaysDate().ISO()

        index = dataConvention.index
        compounding = dataConvention.ois_compounding

        position = kwargs.get("longOrShort")
        if position is None:
            raise ValueError("longOrShort parameter is missing for RFR Swap")

        notional = kwargs.get("notional", dataConvention.contractual_notional)
        ois_spread = kwargs.get("ois_spread", 0.0)

        return ProductRfrSwap(
            effectiveDate=effectiveDate,
            termOrEnd=termOrEnd,
            index=index,
            fixedRate=values,
            position=position,
            notional=notional,
            ois_spread=ois_spread,
            compounding=compounding,
        )
    

    @classmethod
    def _parseAxis1(cls, axis1: str) -> tuple:
        axis1 = axis1.strip()

        if "x" in axis1.lower():
            parts_raw = axis1.replace("X", "x").split("x")
            parts = [p.strip() for p in parts_raw if p.strip()]

            if len(parts) != 2:
                raise ValueError(f"Invalid axis1: '{axis1}'")

            start_part = parts[0]
            end_part = parts[1]

            try:
                datetime.strptime(start_part, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid effective date: '{start_part}'. Expected format: YYYY-MM-DD")

            try:
                datetime.strptime(end_part, "%Y-%m-%d")
                return (start_part, end_part)
            except ValueError:
                tenor = end_part.upper()
                if len(tenor) >= 2 and tenor[:-1].isdigit() and tenor[-1].isalpha():
                    return (start_part, tenor)
                else:
                    raise ValueError(f"Invalid termination: '{end_part}'. Expected YYYY-MM-DD or tenor like '10Y'")

        tenor = axis1.upper()
        if len(tenor) >= 2 and tenor[:-1].isdigit() and tenor[-1].isalpha():
            return (None, tenor)

        raise ValueError(f"Invalid axis1 format: '{axis1}'. Expected 'date x date', 'date x tenor', or tenor like '10Y'")


    @classmethod
    def _inferCompoundingMethod(cls, dataConvention: DataConvention) -> str:
        if hasattr(dataConvention, "compounding"):
            return dataConvention.compounding.upper()

        if dataConvention.data_type == "RFR FUTURE":
            return "AVERAGE"
        elif dataConvention.data_type == "RFR SWAP":
            return "COMPOUND"

        return "COMPOUND"


