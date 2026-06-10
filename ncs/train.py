import os
import sys
import argparse
from shutil import rmtree
import tensorflow as tf

from model.ncs import NCS
from dataset.data import Data
from utils.config import MainConfig
from global_vars import LOGS_DIR, CHECKPOINTS_DIR
import time


def make_model(config):
    model = NCS(config)
    print("Building model...")
    model.build(input_shape=config.input_shape)
    model.summary()
    print("Compiling model...")
    
    # We read learning_rate straight from your experiment JSON profile
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.experiment.learning_rate)
    model.compile(optimizer=optimizer, run_eagerly=True)
    
    if config.experiment.checkpoint is not None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, config.experiment.checkpoint)
        print(f"Warm starting from checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    return model


def main(config):
    log_dir = os.path.join(LOGS_DIR, config.name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, config.name)
    if os.path.isdir(log_dir) or os.path.isdir(checkpoint_dir):
        timestamp = time.strftime("%Y%m%d%H%M%S")
        log_dir = f"{log_dir}_{timestamp}"
        checkpoint_dir = f"{checkpoint_dir}_{timestamp}"
        print(f"Logs/checkpoints for this experiment already exist. Saving to new directories with timestamp: {timestamp}")

    print("Initializing model...")
    model = make_model(config)

    print("Reading data...")
    data = Data(config, mode="train")
    validation_data = Data(config, mode="validation")

    # DYNAMIC STEP CALCULATION
    # len(data) returns the exact number of batches/steps your generator yields per epoch
    batches_per_epoch = len(data)
    save_step_frequency = batches_per_epoch * 1000

    print(f"Detected {batches_per_epoch} steps per epoch. Checkpoints will save every {save_step_frequency} iterations.")

    print("Training...")
    model.fit(
        data,
        validation_data=validation_data,
        epochs=config.experiment.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                write_graph=False,
                write_steps_per_second=False,
                update_freq="epoch",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_m/Loss",
                factor=0.5,
                patience=15,
                mode="min",
                min_lr=1e-6,
                verbose=1
            ),
            # NEW: Best-Only Checkpoint Configuration
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "best_model.weights.h5"), 
                monitor="val_m/Loss",       # Track validation physical energy
                mode="min",                 # Lower energy is better
                save_best_only=True,        # Only write to disk when performance improves
                save_weights_only=True
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    opts = parser.parse_args()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

    # Limit VRAM usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")
        sys.exit()

    config = MainConfig(opts.config)
    main(config)