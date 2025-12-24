import numpy as np
from typing import List
from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.market import *
from fixedincomelib.model import *
from fixedincomelib.product import *
from fixedincomelib.utilities import *
from fixedincomelib.yield_curve.build_method import YieldCurveBuildMethod

class YieldCurve(Model):

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
            gradient_vector += grad
        else:
            gradient_vector[:] = grad

    # def forward_rate(self, effective_date : Date, termination_date : Date):
    #     day_counter : ql.DayCounter = self.component_identifier.dayCounter()
    #     tau = day_counter.yearFraction(effective_date, termination_date)
    #     time_to_effective_date = accrued(self.value_date, effective_date)
    #     time_to_termination_date = accrued(self.value_date, termination_date)
    #     exponent = self.state_data_interpolator.integrate(
    #         time_to_effective_date,
    #         time_to_termination_date
    #     )
    #     forward_rate = 1. / tau * (np.exp(exponent) - 1)

    @property
    def state_data_interpolator(self) -> Interpolator1D:
        return self.interpolator_
    
