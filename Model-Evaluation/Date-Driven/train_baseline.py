"""
Training Pipeline for Data-Driven Baseline Model.

This script executes the training of the baseline BiLSTM model. It relies purely 
on the MSE loss between predicted and actual power, serving as a control experiment
to benchmark the efficacy of physics-informed approaches.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Internal Imports
from config_setup import CONFIG, logger
from network_architecture import DataDrivenBiLSTMBuilder

# --- Data Processing Utilities ---

class DataProcessor:
    """Handles loading, scaling, and sequence generation."""
    
    def __init__(self, filepath, window_size):
        self.filepath = filepath
        self.window_size = window_size
        self.scaler_in = MinMaxScaler()
        self.scaler_out = MinMaxScaler()

    def load_and_preprocess(self):
        if not os.path.exists(self.filepath):
            logger.error(f"Data file not found: {self.filepath}")
            raise FileNotFoundError
            
        logger.info(f"Loading data from {self.filepath}...")
        df = pd.read_csv(self.filepath)
        
        # Extract features and targets
        v_x = df['Vx (m/s)'].values.astype(np.float32)
        v_y = df['Vy (m/s)'].values.astype(np.float32)
        v_z = df['Vz (m/s)'].values.astype(np.float32)
        p_actual = df['Power_filtered'].values.astype(np.float32)
        
        # Derived features
        v_h = np.sqrt(v_x**2 + v_y**2)
        
        # Prepare arrays
        X = np.stack([v_h, v_z], axis=1) # [Samples, 2]
        y = p_actual.reshape(-1, 1)      # [Samples, 1]
        
        # Scaling
        X_scaled = self.scaler_in.fit_transform(X)
        y_scaled = self.scaler_out.fit_transform(y)
        
        # Sequence Generation
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        return X_seq, y_seq

    def _create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.window_size):
            Xs.append(X[i : i + self.window_size])
            ys.append(y[i : i + self.window_size])
        return np.array(Xs), np.array(ys)

# --- Loss Function ---

@tf.function
def compute_loss(model, x, y_true):
    """
    Computes the standard MSE loss + L2 Regularization.
    No physics terms are included here.
    """
    y_pred = model(x, training=True)
    
    # MSE Loss
    loss_mse = tf.reduce_mean(tf.square(y_pred - y_true))
    
    # L2 Regularization on weights
    loss_reg = tf.add_n([tf.nn.l2_loss(w) for w in model.trainable_variables])
    
    total_loss = CONFIG.data_loss_weight * loss_mse + CONFIG.alpha_l2 * loss_reg
    return total_loss

# --- Training Loop ---

def train_model(model, train_ds, val_ds):
    optimizer = tf.keras.optimizers.Adam(CONFIG.learning_rate)
    
    best_val_loss = float('inf')
    patience_cnt = 0
    best_weights = None
    
    history = {'train_loss': [], 'val_loss': []}
    
    logger.info("Starting training...")
    
    for epoch in range(1, CONFIG.epochs + 1):
        # Training Step
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x_b, y_b in train_ds:
            with tf.GradientTape() as tape:
                loss = compute_loss(model, x_b, y_b)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
            
        train_loss = epoch_loss_avg.result().numpy()
        
        # Validation Step
        val_loss_avg = tf.keras.metrics.Mean()
        for x_b, y_b in val_ds:
            loss = compute_loss(model, x_b, y_b)
            val_loss_avg.update_state(loss)
            
        val_loss = val_loss_avg.result().numpy()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            best_weights = model.get_weights()
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG.patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                model.set_weights(best_weights)
                break
                
    return history

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Prepare Data
    processor = DataProcessor(CONFIG.train_data_path, CONFIG.window_size)
    try:
        X, y = processor.load_and_preprocess()
    except FileNotFoundError:
        exit()

    # Split Data (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Create TF Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(CONFIG.batch_size, drop_remainder=True)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(CONFIG.batch_size, drop_remainder=True)
    
    # 2. Build Model
    model = DataDrivenBiLSTMBuilder.build_model(input_shape=(CONFIG.window_size, 2))
    
    # 3. Train
    hist = train_model(model, train_ds, val_ds)
    
    # 4. Save
    save_path = os.path.join(CONFIG.save_dir, CONFIG.model_name)
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # 5. Evaluate (Simple metric check)
    y_pred_scaled = model.predict(X_test, batch_size=CONFIG.batch_size)
    # Simple flatten for metric calc (ignoring sequence reconstruction for brevity here)
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred_scaled.flatten()
    
    # Inverse scale (using dummy reshaping)
    y_test_phys = processor.scaler_out.inverse_transform(y_test_flat.reshape(-1, 1))
    y_pred_phys = processor.scaler_out.inverse_transform(y_pred_flat.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_phys, y_pred_phys)
    r2 = r2_score(y_test_phys, y_pred_phys)
    
    logger.info(f"Test MAE: {mae:.4f} W")
    logger.info(f"Test R2 : {r2:.4f}")
