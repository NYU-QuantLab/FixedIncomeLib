from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.market import *
from fixedincomelib.model import *
from fixedincomelib.yield_curve.build_method import YieldCurveBuildMethod

# class YieldCurveFactory:

    

#     ### api 1
#     @staticmethod
#     def create_model_yield_curve(
#         value_date : Date,
#         data_collection : DataCollection, 
#         build_method_collection : BuildMethodColleciton):

#         build_methods = YieldCurveFactory.\
#             sort_and_flatten_build_method_collection(build_method_collection)
#         for build_method in build_methods:
#             build_method.interpolation
#             data_types = build_method.calibration_instruments()
#             mkt_data = []
#             for data_type in data_types:
#                 data_conv = build_method[data_type]
#                 mkt_data.append(data_collection.get_data_from_data_collection(
#                                 data_type, data_conv))
            
    
#     @staticmethod
#     def calibrate_single_component(
#         value_date : Date, 

#     )

#     ### utils
#     @staticmethod
#     def sort_and_flatten_build_method_collection(
#         build_method_collection : BuildMethodColleciton):
#         # TODO: sort out dependency
#         ordered_bm_list = []
#         for _, bm in build_method_collection.items:
#             ordered_bm_list.append(bm)
#         return ordered_bm_list

# class YieldCurveModelComponent(ModelComponent):

#     def __init__(self, 
#                  value_date : Date, 
#                  calibration_data : DataCollection, 
#                  build_method : YieldCurveBuildMethod) -> None:
        
#         super().__init__(value_date, calibration_data, build_method)
#         interpolation_method = build_method.interpolation_method
#         extrapolation_method = build_method.extrapolation_method
        
#         self.axis1 = []
#         self.timeToDate = []
#         self.ifrInterpolator = None
#         self.targetIndex_ = None
#         self.isOvernightIndex_ = False
#         # i don't like this implementation
#         if '1B' in self.target_: 
#             self.targetIndex_ = IndexRegistry().get(self.target_)
#             self.isOvernightIndex_ = True
#         else:
#             tokenizedIndex = self.target_.split('-')
#             tenor = tokenizedIndex[-1]
#             self.targetIndex_ = IndexRegistry().get('-'.join(tokenizedIndex[:-1]), tenor)
#         self.calibrate()

#     def calibrate(self):
#         ### TODO: calibration to market instruments instead of directly feeding ifr
#         this_df = self.dataCollection_[self.dataCollection_['INDEX'] == self.target_]
#         ### TODO: axis1 can be a combination of dates and tenors
#         ###       for now, i assume they're all tenor based

#         cal = self.targetIndex_.fixingCalendar()
#         for each in this_df['AXIS1'].values.tolist():
#             this_dt = Date(cal.advance(self.valueDate_, Period(each), self.targetIndex_.businessDayConvention()))
#             self.axis1.append(this_dt)
#             self.timeToDate.append(accrued(self.valueDate_, this_dt))
#         self.stateVars = this_df['VALUES'].values.tolist()
#         self.ifrInterpolator = Interpolator1D(self.timeToDate, self.stateVars, self.interpolationMethod_)
    
#     def getStateVarInterpolator(self):
#         return self.ifrInterpolator

#     @property
#     def isOvernightIndex(self):
#         return self.isOvernightIndex_
    
#     @property
#     def targetIndex(self):
#         return self.targetIndex_