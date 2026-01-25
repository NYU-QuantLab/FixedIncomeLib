import numpy as np
from typing import List
from scipy.linalg import block_diag

from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.market import *
from fixedincomelib.model import *
from fixedincomelib.product import *
from fixedincomelib.utilities import *
from fixedincomelib.valuation import *
from fixedincomelib.yield_curve.build_method import YieldCurveBuildMethod
from fixedincomelib.valuation.valuation_engine import ValuationRequest

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

    def calculate_model_jacobian(self):
        super().calculate_model_jacobian()
        # WARNING: WE DO NOT ALLOW A MIXTURE OF CALIBRATION INSTRUMENTS AND STATE DATA FOR NOW
        only_state_data = False
        jacobian_pre = [None] * self.num_components
        for target_name, yc_component in self.components_.items():
            index = self.component_indices[target_name]
            calib_prod = yc_component.calibration_product
            calib_funding = yc_component.calibration_funding
            if len(calib_prod) == 0:
                # no calibration, just using state data
                # jacobian is identity
                jacobian_pre[index] = np.diag(np.ones(yc_component.num_state_data))
                only_state_data = True
                continue
            # calculate calibration instrument gradient
            grads = []
            for _, (prod, funding) in enumerate(zip(calib_prod, calib_funding)):
                fi_vp = FundingIndexParameter({'Funding Index' : funding})
                vpc = ValuationParametersCollection([fi_vp])
                engine = ValuationEngineProductRegistry.new_valuation_engine(
                    self, prod, vpc, ValuationRequest.PV_DETAILED)
                engine.calculate_value()
                grads.append(engine.grad_at_par())
            jacobian_pre[index] = np.concatenate(grads, axis=0)

        jacobian = jacobian = block_diag(*jacobian_pre) if only_state_data else np.concatenate(jacobian_pre, axis=0)
        self.is_jacobian_calculated_ = True
        return jacobian
        
class YieldCurveModelComponent(ModelComponent):

    def __init__(self, 
                 value_date : Date,
                 component_identifier : ql.Index,
                 state_data : np.ndarray,
                 build_method : YieldCurveBuildMethod,
                 calibration_product : Optional[List[Product]]=[],
                 calibration_funding : Optional[List[Product]]=[]) -> None:
        
        super().__init__(value_date, component_identifier, state_data, build_method, calibration_product, calibration_funding)
        assert len(state_data) == 2
        self.num_state_data_ = len(state_data[0])
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            state_data[0], 
            state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method)

    def perturb_model_parameter(self, parameter_id : int, perturb_size : float, override_parameter : Optional[bool]=False):
        super().perturb_model_parameter(parameter_id, perturb_size, override_parameter)
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            self.state_data[0], 
            self.state_data[1],
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
    
    @property
    def num_state_data(self) -> int:
        return self.num_state_data_

### registry
ModelDeserializerRegistry().register(YieldCurve._model_type.to_string(), YieldCurve.deserialize)