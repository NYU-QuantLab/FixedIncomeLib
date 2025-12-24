import numpy as np
from typing import List
from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.market import *
from fixedincomelib.model import *
from fixedincomelib.product import *
from fixedincomelib.utilities import *
from fixedincomelib.yield_curve.build_method import (YieldCurveBuildMethod)
from fixedincomelib.yield_curve.yield_curve_model import (YieldCurve, YieldCurveModelComponent)

class YieldCurveBuilder:

    ### api 1
    @staticmethod
    def create_model_yield_curve(
        value_date : Date,
        data_collection : DataCollection, 
        build_method_collection : BuildMethodColleciton):

        # create the skelton
        model_yield_curve = YieldCurve(value_date, data_collection, build_method_collection)

        # loop through build methods and build component one by one
        build_methods = YieldCurveBuilder.\
            sort_and_flatten_build_method_collection(build_method_collection)
        for i in range(len(build_methods)):
            this_bm : YieldCurveBuildMethod = build_methods[i]
            data_conv_ifr = this_bm.instantaneous_forward_rate
            if data_conv_ifr is not None:
                state_data = data_collection.get_data_from_data_collection(
                    'INSTANTANEOUS FORWARD RATE', data_conv_ifr.name)
                component = YieldCurveBuilder.calibrate_single_component_from_state_data(
                    value_date, data_conv_ifr, state_data, this_bm)
                model_yield_curve.set_model_component(this_bm.target_index, component)
            else:
                pass
                # build_method.interpolation
                # data_types = build_method.calibration_instruments()
                # mkt_data = []
                # for data_type in data_types:
                #     data_conv = build_method[data_type]
                #     mkt_data.append(data_collection.get_data_from_data_collection(
                #                     data_type, data_conv))
        
        return model_yield_curve

    @staticmethod
    def calibrate_single_component_from_state_data(
        value_date : Date,
        data_conv : DataConventionIFR,
        state_data : Data1D,
        build_method : YieldCurveBuildMethod):

        time_to_anchored_dates = []
        values = []
        for i in range(len(state_data.axis1)):
            this_x = state_data.axis1[i]
            if TermOrTerminationDate(this_x).is_term():
                # if it is term
                moved_date = add_period(value_date, 
                                        Period(this_x), 
                                        data_conv.business_day_convention,
                                        data_conv.holiday_convention)
                time = accrued(value_date, moved_date)
            else:
                # if it is date
                time = accrued(value_date, Date(this_x))
            time_to_anchored_dates.append(time)
            values.append(state_data.values[i])

        # check if time instances are sorted
        assert np.all(np.diff(time_to_anchored_dates) >= 0)
        combined_data = np.asarray([time_to_anchored_dates, values])

        return YieldCurveModelComponent(
            value_date,
            build_method.target_index,
            [],
            combined_data,
            build_method)

    ### utils
    @staticmethod
    def sort_and_flatten_build_method_collection(
        build_method_collection : BuildMethodColleciton):
        # TODO: sort out dependency
        ordered_bm_list = []
        for _, bm in build_method_collection.items:
            ordered_bm_list.append(bm)
        return ordered_bm_list