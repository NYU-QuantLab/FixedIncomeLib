import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from fixedincomelib.model.model import Model, ModelComponent
from fixedincomelib.date import Date
from fixedincomelib.utilities.numerics import Interpolator2D
from fixedincomelib.yield_curve import YieldCurve
from fixedincomelib.data import DataCollection, Data2D, Data1D
from fixedincomelib.data import build_yc_data_collection

class SabrModel(Model):
    MODEL_TYPE = "IR_SABR"
    PARAMETERS = ["NORMALVOL", "BETA", "NU", "RHO"]
    
    def __init__(
        self,
        valueDate: str,
        dataCollection: DataCollection,
        buildMethodCollection: List[Dict[str, Any]],
        ycModel: YieldCurve,
    ):
        for bm in buildMethodCollection:
            tgt  = bm["TARGET"]
            vals = bm["VALUES"]
            prod = bm.get("PRODUCT")
            bm["NAME"] = f"{tgt}-{vals}" + (f"-{prod}" if prod else "")

        super().__init__(valueDate, self.MODEL_TYPE, dataCollection, buildMethodCollection)
        self.subModel_ = ycModel

        # Cache YC param count (avoid computing YC jacobian inside weights mapping)
        self._n_yc_params = int(np.asarray(self.subModel_.getGradientArray()).size)
        self._build_sabr_layout()

        n_total = int(self._n_yc_params + self._num_sabr_pillars)
        self.gradient_ = np.zeros(n_total, dtype=float)

        self._build_gradient_labels()

    @classmethod
    def from_curve(
        cls,
        valueDate: str,
        dataCollection: DataCollection,
        buildMethodCollection: List[Dict[str, Any]],
        ycModel: YieldCurve
    ) -> "SabrModel":
        return cls(valueDate, dataCollection, buildMethodCollection, ycModel)

    # @classmethod
    # def from_data(
    #     cls,
    #     valueDate: str,
    #     dataCollection: DataCollection,
    #     buildMethodCollection: List[Dict[str, Any]],
    #     ycData: DataCollection,
    #     ycBuildMethods: List[Dict[str, Any]]
    # ) -> "SabrModel":
    #     zero_curves = []
    #     for idx_name, sub in ycData.groupby("INDEX"):
    #         d1 = Data1D.createDataObject(
    #             data_type="zero_rate",x
    #             data_convention=idx_name,
    #             df=sub[["AXIS1", "VALUES"]]
    #         )
    #         zero_curves.append(d1)
    #     yc_dc = DataCollection(zero_curves)

    #     yc = YieldCurve(valueDate, yc_dc, ycBuildMethods)
    #     return cls(valueDate, dataCollection, buildMethodCollection, yc)

    @classmethod
    def from_data(
        cls,
        valueDate: str,
        dataCollection: DataCollection,                 # SABR surfaces (Data2D)
        buildMethodCollection: List[Dict[str, Any]],     # SABR build methods
        yc_market_df,                                   # calibration instrument quotes
        ycBuildMethods: List[Dict[str, Any]]
    ) -> "SabrModel":
        
        _, yc_dc = build_yc_data_collection(yc_market_df)
        yc = YieldCurve(valueDate, yc_dc, ycBuildMethods)
        return cls(valueDate, dataCollection, buildMethodCollection, yc)

    def newModelComponent(self, build_method: Dict[str, Any]) -> ModelComponent:
        return SabrModelComponent(self.valueDate, self.dataCollection, build_method)

    def get_sabr_parameters(
        self,
        index: str,
        expiry: float,
        tenor: float,
        product_type: str | None = None
    ) -> Tuple[float, float, float, float, float, float]:
        suffix = f"-{product_type}".upper() if product_type else ""
        params = []
        for p in self.PARAMETERS:
            key = f"{index}-{p}{suffix}".upper()
            comp = self.components.get(key)
            if comp is None:
                raise KeyError(f"No SABR component found for {key}")
            params.append(comp.interpolate(expiry, tenor))
        nv_key = f"{index}-NORMALVOL{suffix}".upper()
        nv_comp = self.components[nv_key]
        return (*params, nv_comp.shift, nv_comp.vol_decay_speed)
    
    # Layout for SABR pillars
    def _build_sabr_layout(self) -> None:
        """
        Defines the ordering of SABR pillar parameters within the SABR block.

        Ordering is:
            - components in insertion order of self.components
            - within a component, row-major flatten of the surface grid
        """
        self._sabr_slice_by_key: Dict[str, slice] = {}
        offset = 0
        for key, comp in self.components.items():  # insertion-ordered dict
            n = int(len(getattr(comp, "stateVars_", [])))
            if n == 0:
                # fallback if stateVars_ isn't set (should not happen if component is implemented properly)
                n = int(np.asarray(getattr(comp, "grid")).size)
            self._sabr_slice_by_key[key] = slice(offset, offset + n)
            offset += n
        self._num_sabr_pillars = int(offset)

    def num_sabr_pillars(self) -> int:
        return int(self._num_sabr_pillars)

    def sabr_component_key(self, index: str, parameter: str, product_type: str | None = None) -> str:
        suffix = f"-{product_type}".upper() if product_type else ""
        return f"{index}-{parameter}{suffix}".upper()

    def sabr_global_node_indices_and_weights(
        self,
        index: str,
        parameter: str,
        expiry: float,
        tenor: float,
        product_type: str | None = None,
    ) -> List[Tuple[int, float]]:
        """
        For a given SABR surface (index + parameter + optional product_type),
        return the *global model parameter indices* and interpolation weights
        for the 4 pillars used at (expiry, tenor).

        Global model parameter ordering:
            [YC params] + [SABR params]
        """
        key = self.sabr_component_key(index, parameter, product_type)
        comp = self.components.get(key)
        if comp is None:
            raise KeyError(f"No SABR component found for {key}")

        # local weights inside the component: (flat_idx, w)
        local = comp.weights(expiry, tenor)

        # global offset = YC param count + SABR component slice start
        n_yc = int(self._n_yc_params)

        sl = self._sabr_slice_by_key[key]
        base = int(sl.start)

        return [(n_yc + base + int(flat_idx), float(w)) for flat_idx, w in local]
    
    def jacobian(self) -> np.ndarray:
        """
            J = [[ J_yc, 0 ],
                 [  0 ,  I ]]

        - J_yc: yield curve calibration Jacobian
        - I: identity for SABR pillars (no SABR calibration in-code)
        """
        J_yc = np.asarray(self.subModel_.jacobian(), dtype=float)
        if J_yc.ndim != 2 or J_yc.shape[0] != J_yc.shape[1]:
            raise ValueError(f"YieldCurve jacobian must be square, got shape {J_yc.shape}")

        n = int(J_yc.shape[0])
        m = int(self.num_sabr_pillars())

        J = np.zeros((n + m, n + m), dtype=float)
        J[:n, :n] = J_yc
        if m > 0:
            J[n:, n:] = np.eye(m, dtype=float)
        return J

    def calibration_quote_sensitivity(self) -> np.ndarray:
        """
        Quote sensitivity scaling used by risk_reporting.py:
            quote_risk = S * param_risk

        - YC part uses YieldCurve.calibration_quote_sensitivity()
        - SABR pillars: quote == parameter => sensitivity = 1 for each pillar
        """
        S_yc = np.asarray(self.subModel_.calibration_quote_sensitivity(), dtype=float)
        m = int(self.num_sabr_pillars())
        if m == 0:
            return S_yc
        return np.concatenate([S_yc, np.ones(m, dtype=float)])
    
    #Gradient
    def clearGradient(self) -> None:
        self.gradient_[:] = 0.0

    def getGradientArray(self) -> np.ndarray:
        return np.asarray(self.gradient_, dtype=float).copy()

    def _build_gradient_labels(self) -> None:
        """
        Optional: create labels so the gradient vector is human-readable.
        """
        labels: List[str] = []

        # YC labels if available, else fallback
        yc_labels = getattr(self.subModel_, "gradient_labels_", None)
        if yc_labels is None:
            yc_labels = [f"YC[{i}]" for i in range(int(self._n_yc_params))]
        labels.extend(list(yc_labels))

        # SABR pillar labels
        for key, sl in self._sabr_slice_by_key.items():
            n = int(sl.stop - sl.start)
            labels.extend([f"{key}[{k}]" for k in range(n)])

        self.gradient_labels_ = labels
    
    @property
    def subModel(self):
        return self.subModel_
    
class SabrModelComponent(ModelComponent):

    def __init__(
        self,
        valueDate: Date,
        dataCollection: DataCollection,
        buildMethod: Dict[str, Any]
    ) -> None:
        
        super().__init__(valueDate, dataCollection, buildMethod)
        self.shift           = float(buildMethod.get("SHIFT", 0.0))
        self.vol_decay_speed = float(buildMethod.get("VOL_DECAY_SPEED", 0.0))
        self.product_type    = buildMethod.get("PRODUCT")

        self.axis1: np.ndarray
        self.axis2: np.ndarray
        self.grid: np.ndarray
        self._shape: Tuple[int, int]

        self._interp2d: Interpolator2D

        self.calibrate()

    def calibrate(self) -> None:

        param = self.buildMethod_["VALUES"]  

        md = self.dataCollection.get(param.lower(), self.target_)
        assert isinstance(md, Data2D)

        self.axis1 = np.array(md.axis1, dtype=float)  
        self.axis2 = np.array(md.axis2, dtype=float)   
        self.grid = np.array(md.values, dtype=float)

        if self.grid.shape != (len(self.axis1), len(self.axis2)):
            raise ValueError(f"{self.target_}: grid shape {self.grid.shape} "f"!= ({len(self.axis1)}, {len(self.axis2)})")

        self._shape = self.grid.shape  

        # Make grid nodes the internal parameter vector (row-major flatten)
        self.stateVars_ = self.grid.reshape(-1).astype(float).tolist()                      

        method = str(self.buildMethod_.get("INTERPOLATION", "LINEAR")).upper()
        if method != "LINEAR":
            raise ValueError(f"{self.target_}: only LINEAR interpolation supported for risk, got {method}")
        self._interp2d = Interpolator2D(
            axis1=self.axis1,
            axis2=self.axis2,
            values=self.grid,
            method=method
        )

    def interpolate(self, expiry: float, tenor: float) -> float:
        return self._interp2d.interpolate(expiry, tenor)
    
    def _flat_index(self, i: int, j: int) -> int:
        return int(i) * int(len(self.axis2)) + int(j)

    def weights(self, expiry: float, tenor: float):
        x = float(expiry)
        y = float(tenor)

        ax1 = np.asarray(self.axis1, dtype=float)
        ax2 = np.asarray(self.axis2, dtype=float)

        n1 = len(ax1)
        n2 = len(ax2)

        if n1 == 0 or n2 == 0:
            raise ValueError("Empty SABR surface axes.")

        # ---- Degenerate cases: 1D/0D surfaces ----
        if n1 == 1 and n2 == 1:
            # only one node
            return [(0, 1.0)]

        if n2 == 1:
            # expiry-only interpolation, tenor ignored (surface constant in tenor)
            if n1 == 1:
                return [(0, 1.0)]

            i = int(np.searchsorted(ax1, x) - 1)
            i = max(0, min(i, n1 - 2))

            x1, x2 = float(ax1[i]), float(ax1[i + 1])
            if x2 == x1:
                # degenerate: collapse to nearest
                ii = i if abs(x - x1) <= abs(x - x2) else i + 1
                return [(ii * n2 + 0, 1.0)]

            w2 = (x - x1) / (x2 - x1)
            w2 = max(0.0, min(1.0, w2))
            w1 = 1.0 - w2

            # flat index = i*n2 + 0 (since n2==1)
            return [(i * n2 + 0, w1), ((i + 1) * n2 + 0, w2)]

        if n1 == 1:
            # tenor-only interpolation, expiry ignored (surface constant in expiry)
            j = int(np.searchsorted(ax2, y) - 1)
            j = max(0, min(j, n2 - 2))

            y1, y2 = float(ax2[j]), float(ax2[j + 1])
            if y2 == y1:
                jj = j if abs(y - y1) <= abs(y - y2) else j + 1
                return [(0 * n2 + jj, 1.0)]

            w2 = (y - y1) / (y2 - y1)
            w2 = max(0.0, min(1.0, w2))
            w1 = 1.0 - w2

            return [(0 * n2 + j, w1), (0 * n2 + (j + 1), w2)]

        # ---- Standard 2D bilinear interpolation (your existing logic) ----
        i = int(np.searchsorted(ax1, x) - 1)
        j = int(np.searchsorted(ax2, y) - 1)

        i = max(0, min(i, n1 - 2))
        j = max(0, min(j, n2 - 2))

        x1, x2 = float(ax1[i]), float(ax1[i + 1])
        y1, y2 = float(ax2[j]), float(ax2[j + 1])

        # guards
        if x2 == x1 and y2 == y1:
            return [(i * n2 + j, 1.0)]
        if x2 == x1:
            # linear in y only
            wy2 = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
            wy2 = max(0.0, min(1.0, wy2))
            wy1 = 1.0 - wy2
            return [(i * n2 + j, wy1), (i * n2 + (j + 1), wy2)]
        if y2 == y1:
            # linear in x only
            wx2 = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
            wx2 = max(0.0, min(1.0, wx2))
            wx1 = 1.0 - wx2
            return [(i * n2 + j, wx1), ((i + 1) * n2 + j, wx2)]

        # bilinear
        tx = (x - x1) / (x2 - x1)
        ty = (y - y1) / (y2 - y1)
        tx = max(0.0, min(1.0, tx))
        ty = max(0.0, min(1.0, ty))

        w11 = (1 - tx) * (1 - ty)
        w21 = tx * (1 - ty)
        w12 = (1 - tx) * ty
        w22 = tx * ty

        return [
            (i * n2 + j, w11),
            ((i + 1) * n2 + j, w21),
            (i * n2 + (j + 1), w12),
            ((i + 1) * n2 + (j + 1), w22),
        ]
    
    def _sync_from_statevars(self) -> None:
        self.grid = np.asarray(self.stateVars_, dtype=float).reshape(self._shape)
        self._interp2d = Interpolator2D(
            axis1=self.axis1,
            axis2=self.axis2,
            values=self.grid,
            method="LINEAR"
        )

    def perturbModelParameter(self, state_var_index: int, perturb_size: float) -> None:
        super().perturbModelParameter(state_var_index, perturb_size)
        self._sync_from_statevars()
