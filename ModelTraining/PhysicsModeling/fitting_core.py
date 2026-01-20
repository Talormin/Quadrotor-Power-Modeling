# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from physics_models import UAVPowerModels

class OrthogonalSparseFitter:
    def __init__(self):
        self.popt_h = None
        self.popt_v = None
        self.P_hover_opt = None
        self.sparse_coeffs = None
        self.residual_vector = None

    def fit_physics_base(self, Vh, Ph, Vv_combined, Pv_combined, p0_v, bounds_v):
        """
        Step 1: 拟合基础物理模型 (Horizontal & Vertical)
        """
        # 水平拟合
        self.popt_h, _ = curve_fit(
            UAVPowerModels.horizontal_model, Vh, Ph, maxfev=20000
        )
        # 垂直拟合
        self.popt_v, _ = curve_fit(
            UAVPowerModels.unified_vertical_model, 
            Vv_combined, Pv_combined, 
            p0=p0_v, bounds=bounds_v, maxfev=30000
        )
        return self.popt_h, self.popt_v

    def fit_hover_offset(self, vx, vy, vz, P_actual):
        """
        Step 2: 优化悬停功率偏置
        """
        V_h = np.sqrt(vx**2 + vy**2)
        
        # 定义内部损失函数
        def _loss_hover(P_hover_val):
            # 计算基础项
            P_h_part = UAVPowerModels.horizontal_model(V_h, *self.popt_h)
            P_v_part = UAVPowerModels.unified_vertical_model(vz, *self.popt_v)
            
            # 扣除常数项，防止重复计算
            offset_h = P_h_part - self.popt_h[0]
            offset_v = P_v_part - self.popt_v[0]
            
            P_pred = P_hover_val + offset_h + offset_v
            
            # MAPE Loss
            return np.mean(np.abs((P_actual - P_pred) / np.clip(P_actual, 1e-8, None)))

        res = minimize_scalar(_loss_hover, bounds=(0, 400), method='bounded')
        self.P_hover_opt = float(res.x)
        return self.P_hover_opt

    def fit_orthogonal_sparse_residual(self, vx, vy, vz, P_actual):
        """
        Step 3: 正交投影 + 稀疏回归 (SINDy thought process)
        """
        V_h = np.sqrt(vx**2 + vy**2)
        
        # 1. 构造基础项向量 Base Vectors
        H_vec = UAVPowerModels.horizontal_model(V_h, *self.popt_h) - self.popt_h[0]
        V_vec = UAVPowerModels.unified_vertical_model(vz, *self.popt_v) - self.popt_v[0]
        ones = np.ones_like(H_vec)
        
        # 基础模型预测值
        y_base = self.P_hover_opt * ones + H_vec + V_vec
        
        # 2. 正交投影 (Orthogonal Projection)
        # B = [1, H, V]
        B = np.c_[ones, H_vec, V_vec].astype(np.float64)
        Q, R = np.linalg.qr(B, mode='reduced')
        
        # 计算 Q 的有效秩
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-10))
        Q = Q[:, :rank]

        def _project_perp(X):
            return X - Q @ (Q.T @ X)

        # 对目标残差和特征库进行投影
        y_target_perp = _project_perp(P_actual - y_base)
        Phi_features = UAVPowerModels.generate_candidate_features(V_h, vz)
        Phi_perp = _project_perp(Phi_features)

        # 3. LassoCV 稀疏回归
        scaler = StandardScaler()
        Phi_std = scaler.fit_transform(Phi_perp)
        
        lasso = LassoCV(cv=5, fit_intercept=False, max_iter=30000, n_alphas=100, random_state=42)
        lasso.fit(Phi_std, y_target_perp)
        
        # 恢复系数
        std_scale = scaler.scale_
        valid_mask = std_scale > 1e-8
        
        self.sparse_coeffs = np.zeros(Phi_features.shape[1])
        if np.any(valid_mask):
            self.sparse_coeffs[valid_mask] = lasso.coef_ / std_scale[valid_mask]

        # 计算最终余项 r (在正交补空间中)
        self.residual_vector = Phi_perp @ self.sparse_coeffs
        
        return y_base + self.residual_vector
