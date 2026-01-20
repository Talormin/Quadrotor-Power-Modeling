"""
Physics Constraints and Equation Definitions for UAV Power Modeling.

This module defines the underlying physical laws governing the power consumption
of Quadrotor UAVs. It includes aerodynamic coefficient management and
differentiable TensorFlow implementations of power equations.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Union, Tuple, Optional

class AerodynamicConstants:
    """
    Singleton class managing aerodynamic coefficients derived from 
    wind tunnel experiments and sparse regression analysis.
    """
    _instance = None

    # Coefficients C1-C5 for Horizontal Flight
    C1: float = 537.92430435
    C2: float = -11.81444764
    C3: float = -32.51778232
    C4: float = 1851.19680972
    C5: float = 2.00966979

    # Coefficients C6-C9 for Vertical Flight
    C6: float = 2.77965346e+02
    C7: float = 3.40071433e+01
    C8: float = 2.08030427e-01
    C9: float = 4.00000000e+00

    # Base Hovering Power
    P_HOVER: float = 337.09

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AerodynamicConstants, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_params(cls) -> Dict[str, float]:
        """Return all coefficients as a dictionary."""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and isinstance(v, (int, float))}


class PhysicsEquations:
    """
    TensorFlow-based implementations of UAV power consumption equations.
    Designed to be differentiable for Physics-Informed Neural Network (PINN) training.
    """

    @staticmethod
    @tf.function
    def compute_horizontal_power(v_h: tf.Tensor) -> tf.Tensor:
        """
        Computes the theoretical power consumption for horizontal flight.
        
        Equation:
            P_h = C1 + C2*v^2 + C3*sqrt(1 + v^4/C4 - v^2/C5) + C5*v^3
        
        Args:
            v_h (tf.Tensor): Horizontal velocity magnitude (m/s).
            
        Returns:
            tf.Tensor: Estimated power component (Watts), centered by removing C1.
        """
        # Ensure numerical stability with epsilon
        EPSILON = 1e-6
        consts = AerodynamicConstants()
        
        term_inner = 1.0 + (tf.pow(v_h, 4) / consts.C4) - (tf.pow(v_h, 2) / consts.C5)
        # Apply ReLU-like protection to prevent NaN gradients in sqrt
        term_inner_stable = tf.maximum(term_inner, EPSILON)
        
        p_h = (consts.C1 + 
               consts.C2 * tf.pow(v_h, 2) + 
               consts.C3 * tf.sqrt(term_inner_stable) + 
               consts.C5 * tf.pow(v_h, 3))
        
        # Return the dynamic component (offset from base hover)
        return p_h - consts.C1

    @staticmethod
    @tf.function
    def compute_vertical_power(v_v: tf.Tensor) -> tf.Tensor:
        """
        Computes the theoretical power consumption for vertical maneuvers (Ascent/Descent).
        
        Uses a piecewise function based on momentum theory:
        - Ascent (v_v > 0)
        - Descent (v_v <= 0)
        
        Args:
            v_v (tf.Tensor): Vertical velocity (m/s), Up is positive.
            
        Returns:
            tf.Tensor: Estimated vertical power component (Watts).
        """
        EPSILON = 1e-6
        consts = AerodynamicConstants()
        
        # Pre-compute ratio terms
        ratio_89 = 4.0 * consts.C8 / consts.C9
        ratio_79 = 4.0 * consts.C7 / consts.C9
        
        # Ascent Calculation
        inner_up = (1.0 + ratio_89) * tf.pow(v_v, 2) + ratio_79
        sqrt_up = tf.sqrt(tf.maximum(inner_up, EPSILON))
        val_up = (consts.C6 + 
                  consts.C7 * v_v + 
                  consts.C8 * tf.pow(v_v, 3) + 
                  (consts.C7 + consts.C8 * tf.pow(v_v, 2)) * sqrt_up)

        # Descent Calculation
        inner_down = (1.0 - ratio_89) * tf.pow(v_v, 2) + ratio_79
        sqrt_down = tf.sqrt(tf.maximum(inner_down, EPSILON))
        val_down = (consts.C6 + 
                    consts.C7 * v_v - 
                    consts.C8 * tf.pow(v_v, 3) + 
                    (consts.C7 - consts.C8 * tf.pow(v_v, 2)) * sqrt_down)

        # Conditional execution based on direction
        p_v = tf.where(v_v > 0, val_up, val_down)
        
        return p_v - consts.C6


class HybridLossCalculator:
    """
    Computes the composite loss function for PI-S-LSTM training.
    Combines Data-Driven Loss (MSE) with Physics-Informed Residuals.
    """

    def __init__(self, 
                 lambda_data: float = 1.0, 
                 lambda_phy: float = 0.1, 
                 lambda_smooth: float = 1e-5,
                 lambda_reg: float = 1e-6):
        """
        Initialize the loss calculator with hyperparameters.
        
        Args:
            lambda_data: Weight for empirical data loss.
            lambda_phy: Weight for physical consistency loss.
            lambda_smooth: Weight for temporal smoothness (1st derivative).
            lambda_reg: Weight for L2 regularization of weights.
        """
        self.weights = {
            'data': lambda_data,
            'phy': lambda_phy,
            'smooth': lambda_smooth,
            'reg': lambda_reg
        }

    @tf.function
    def __call__(self, 
                 model: tf.keras.Model, 
                 x_input: tf.Tensor, 
                 y_true: tf.Tensor, 
                 y_phy_ref: tf.Tensor, 
                 alpha_d: tf.Tensor, 
                 alpha_p: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Calculate total loss and individual components.
        """
        # Forward pass
        y_pred = model(x_input, training=True)
        
        # 1. Empirical Error (MSE)
        # Using reduce_mean on axis=1 to average over time steps first
        mse_data = tf.reduce_mean(tf.square(y_pred - y_true), axis=1)
        
        # 2. Physics Consistency Error
        mse_phy = tf.reduce_mean(tf.square(y_pred - y_phy_ref), axis=1)
        
        # 3. Temporal Smoothness (First-order difference constraint)
        # Penalizes sudden, unrealistic spikes in power prediction
        diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        loss_smooth = tf.reduce_mean(tf.square(diff_pred))
        
        # 4. Model Weight Regularization (L2)
        loss_reg = tf.add_n([tf.nn.l2_loss(w) for w in model.trainable_variables])
        
        # 5. Adaptive Weighting Mechanism
        # Adjusts influence based on reliability scores (Alpha)
        # Centering heuristics: D - 0.1, P + 0.1
        w_d = tf.reduce_mean(alpha_d, axis=1, keepdims=True) - 0.1
        w_p = tf.reduce_mean(alpha_p, axis=1, keepdims=True) + 0.1
        
        # Ensure dimensions match for broadcasting
        if len(mse_phy.shape) == 1:
            mse_phy = tf.expand_dims(mse_phy, axis=-1)

        # Weighted Sum
        weighted_loss = (w_d * self.weights['data'] * mse_data + 
                         w_p * self.weights['phy'] * mse_phy)
        
        total_loss = (tf.reduce_mean(weighted_loss) + 
                      self.weights['smooth'] * loss_smooth + 
                      self.weights['reg'] * loss_reg)
        
        # Return detailed metrics for logging
        metrics = {
            'loss_data': tf.reduce_mean(mse_data),
            'loss_phy': tf.reduce_mean(mse_phy),
            'loss_smooth': loss_smooth,
            'loss_reg': loss_reg
        }
        
        return total_loss, metrics
