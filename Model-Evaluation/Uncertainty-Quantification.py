"""
Physics-Informed BiLSTM Architecture Definition.

This module explicitly defines the neural network topology. 
Re-instantiating the model structure via code is mandatory for Monte Carlo Dropout,
as it ensures control over the stochastic state of Dropout layers during inference.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import tensorflow as tf

class PhysicsBiLSTMBuilder:
    """
    Constructs the Physics-Informed BiLSTM network with a specific topology
    designed for aerodynamic power estimation.
    """

    @staticmethod
    def build(input_shape=(None, 2)) -> tf.keras.Model:
        """
        Builds the uncompiled Keras model graph.
        
        Architecture:
        Input -> BiLSTM(128) -> BiLSTM(64) -> LayerNorm -> Dense Block -> Residual -> Physics Branching
        
        Args:
            input_shape: Shape tuple (TimeSteps, Features).
            
        Returns:
            tf.keras.Model: The constructed model instance.
        """
        inputs = tf.keras.Input(shape=input_shape, name="kinematic_input")

        # --- Recurrent Feature Extraction ---
        # Layer 1
        fwd_1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(128), return_sequences=True)
        bwd_1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(128), return_sequences=True, go_backwards=True)
        x = tf.keras.layers.Bidirectional(fwd_1, backward_layer=bwd_1, name="bilstm_1")(inputs)
        
        # Layer 2
        fwd_2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64), return_sequences=True)
        bwd_2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64), return_sequences=True, go_backwards=True)
        x = tf.keras.layers.Bidirectional(fwd_2, backward_layer=bwd_2, name="bilstm_2")(x)

        # --- Normalization & Interpretation ---
        x = tf.keras.layers.LayerNormalization(name="layer_norm")(x)
        
        # Dense Block with Regularization
        x = tf.keras.layers.Dense(
            128, activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name="dense_1"
        )(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)
        
        x = tf.keras.layers.Dense(64, activation='relu', name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_2")(x)
        
        x = tf.keras.layers.Dense(64, activation='relu', name="dense_3")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_3")(x)

        # Residual Connection
        residual = tf.keras.layers.Dense(64, name="residual_proj")(inputs)
        x = tf.keras.layers.Add(name="residual_add")([x, residual])

        # --- Physics Branching Output ---
        # Output 4 components: [P_v, P_h, P_hov, P_add]
        phys_outputs = tf.keras.layers.Dense(4, name="physical_components")(x)

        # Summation Layer (Total Power)
        output = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name="output"
        )(phys_outputs)

        return tf.keras.Model(inputs=inputs, outputs=[output, phys_outputs])
