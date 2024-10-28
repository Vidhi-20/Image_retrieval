# src/config.py
class Config:
    DATA_PATH = './data'
    BATCH_SIZE = 64
    EPOCHS = 10
    K_STEPS = 1  # Gibbs Sampling steps in RBM
    FEATURE_SIZE = 512  # Feature vector length for the last layer in DBN
    LEARNING_RATE = 0.01
