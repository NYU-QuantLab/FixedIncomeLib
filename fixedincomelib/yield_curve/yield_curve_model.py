import numpy as np
from typing import List

from yaml import serialize
from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.market import *
from fixedincomelib.model import *
from fixedincomelib.product import *
from fixedincomelib.utilities import *
from fixedincomelib.yield_curve.build_method import YieldCurveBuildMethod

class YieldCurve(Model):

    _version = 1
    _model_type = ModelType.YIELD_CURVE
    

    def __init__(self,
                 value_date : Date,
                 data_collection : DataCollection,
                 build_method_collection : BuildMethodColleciton) -> None:
        super().__init__(value_date, self._model_type, data_collection, build_method_collection)
    
    def discount_factor(self, index : ql.Index, expiry_date : Date):
        this_component : YieldCurveModelComponent = self.retrieve_model_component(index)
        return this_component.discount_factor(expiry_date)

    def discount_factor_gradient_wrt_state(
            self, 
            index : ql.Index,
            expiry_date : Date,
            gradient_vector : List[np.ndarray],
            scaler : Optional[float]=1.,
            accumulate : Optional[bool]=False):
        
        this_component : YieldCurveModelComponent = self.retrieve_model_component(index)
        component_index = self.component_indices[index.name()]
        this_gradient = gradient_vector[component_index]
        this_component.discount_factor_gradient_wrt_state(expiry_date, this_gradient, scaler, accumulate)

    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = YieldCurve._version
        content['MODEL_TYPE'] = YieldCurve._model_type.to_string()
        content['VALUE_DATE'] = self.value_date.ISO()
        content['BUILD_METHOD_COLLECTION'] = self.build_method_collection.serialize()
        content['DATA_COLLECTION'] = self.data_collection.serialize()
        return content

    @classmethod
    def deserialize(cls, input_dict : dict) -> 'YieldCurve':
        input_dict_  = input_dict.copy()
        assert 'VERSION' in input_dict_
        version = input_dict_['VERSION']
        assert 'MODEL_TYPE' in input_dict_
        model_type = input_dict_['MODEL_TYPE']
        assert 'VALUE_DATE' in input_dict_
        value_date = Date(input_dict_['VALUE_DATE'])
        bmc = BuildMethodColleciton.deserialize(input_dict_['BUILD_METHOD_COLLECTION'])
        dc = DataCollection.deserialize(input_dict_['DATA_COLLECTION'])
        # find modelbuilder
        func = ModelBuilderRegistry().get(model_type)
        return func(value_date, dc, bmc)

class YieldCurveModelComponent(ModelComponent):

    def __init__(self, 
                 value_date : Date,
                 component_identifier : ql.Index,
                 calibration_product : list[Product],
                 state_data : np.ndarray,
                 build_method : YieldCurveBuildMethod) -> None:
        
        super().__init__(value_date, component_identifier, calibration_product, state_data, build_method)
        assert len(state_data) == 2
        self.num_state_data_ = len(state_data[0])
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            state_data[0], 
            state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method)

    def discount_factor(self, expiry_date : Date):
        time_to_expiry = accrued(self.value_date, expiry_date)
        exponent = self.state_data_interpolator.integrate(0., time_to_expiry)
        return np.exp(-exponent)

    def discount_factor_gradient_wrt_state(
            self, 
            expiry_date : Date,
            gradient_vector : np.ndarray,
            scaler : Optional[float]=1.,
            accumulate : Optional[bool]=False):
        
        time_to_expiry = accrued(self.value_date, expiry_date)
        exponent = self.state_data_interpolator.integrate(0., time_to_expiry)

        # df risk w.r.t state variables
        d_df_d_exponent = - np.exp(-exponent)
        grad = self.state_data_interpolator.gradient_of_integrated_value_wrt_ordinate(0, time_to_expiry)
        grad *= d_df_d_exponent * scaler

        # finalize
        if accumulate:
            assert len(gradient_vector) == len(grad)
            for i in range(len(gradient_vector)):
                gradient_vector[i] += grad[i]
        else:
            gradient_vector[:] = grad

    @property
    def state_data_interpolator(self) -> Interpolator1D:
        return self.interpolator_
    
### registry
ModelDeserializerRegistry().register(YieldCurve._model_type.to_string(), YieldCurve.deserialize)