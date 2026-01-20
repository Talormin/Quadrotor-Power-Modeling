# -*- coding: utf-8 -*-
import numpy as np

class UAVPowerModels:
    """
    四旋翼无人机功率消耗的物理模型定义
    包含水平飞行、垂直升降及组合特征项
    """

    @staticmethod
    def horizontal_model(v, C1, C2, C3, C4, C5):
        """方程 (20): 水平飞行功率模型"""
        # 防止根号内出现负数的数值保护
        inner = 1 + (v**4)/C4 - (v**2)/C5
        inner = np.maximum(inner, 1e-6)
        return C1 + C2 * v**2 + C3 * np.sqrt(inner) + C5 * v**3

    @staticmethod
    def unified_vertical_model(v, C6, C7, C8, C9):
        """方程 (21): 垂直上升/下降统一模型"""
        v = np.array(v)
        result = np.zeros_like(v)

        # 上升部分 (v > 0)
        up_mask = v > 0
        v_up = v[up_mask]
        inner_up = (1 + 4*C8/C9)*v_up**2 + 4*C7/C9
        # 简单的数值保护，防止极端参数导致的负值
        inner_up = np.maximum(inner_up, 0)
        result[up_mask] = (
            C6 + C7*v_up + C8*v_up**3 +
            (C7 + C8*v_up**2) * np.sqrt(inner_up)
        )

        # 下降部分 (v <= 0)
        down_mask = ~up_mask
        v_down = v[down_mask]
        inner_down = (1 - 4*C8/C9)*v_down**2 + 4*C7/C9
        
        # 仅作为警告，实际运算中进行截断保护
        if np.any(inner_down < 0):
            # print("Warning: Negative value in sqrt for descent phase, clipped to 0.")
            inner_down = np.maximum(inner_down, 0)
            
        result[down_mask] = (
            C6 + C7*v_down - C8*v_down**3 +
            (C7 - C8*v_down**2) * np.sqrt(inner_down)
        )
        return result

    @staticmethod
    def generate_candidate_features(V_h, V_v):
        """
        构建用于稀疏回归的候选特征库 (Library)
        包括高阶项、交叉项及三角函数项
        """
        # 预计算高阶幂
        Vh_sq, Vv_sq = V_h**2, V_v**2
        Vh_cub, Vv_cub = V_h**3, V_v**3
        Vh_quart, Vv_quart = V_h**4, V_v**4
        Vh_quint, Vv_quint = V_h**5, V_v**5

        # 构建特征矩阵 (N, m)
        additional_matrix = np.stack([
            V_h, V_v,
            Vh_sq, Vv_sq,
            Vh_cub, Vv_cub,
            Vh_quart, Vv_quart,
            Vh_quint, Vv_quint,
            V_h * V_v,             # VhVv
            Vh_cub * V_v,          # Vh^3 Vv
            V_h * Vv_cub,          # Vh Vv^3
            Vh_sq * V_v,           # Vh^2 Vv
            Vv_sq * V_h,           # Vv^2 Vh
            Vh_cub * Vv_sq,        # Vh^3 Vv^2
            Vh_sq * Vv_sq,         # Vh^2 Vv^2
            Vh_quart * V_v,        # Vh^4 Vv
            V_h * Vv_quart,        # Vh Vv^4
            Vh_quint * V_v,        # Vh^5 Vv
            V_h * Vv_quint,        # Vh Vv^5
            np.sin(V_h), np.sin(V_v),
        ], axis=1)
        
        return additional_matrix
