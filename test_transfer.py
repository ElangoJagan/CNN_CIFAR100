from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.transfer_learning import TransferModel
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.experiment_tracker import ExperimentTracker

import tensorflow as tf

IMG_SIZE = 96
BATCH_SIZE = 32

# Step 1: Data
loader = DataLoader()
raw_path = loader.initiate_data_loader()

preprocessor = Preprocessor(raw_data_path=raw_path)
x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()

# Step 2: Build tf.data pipelines -- resize happens per-batch, not all at once
def resize_fn(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Step 3: Build transfer model
transfer = TransferModel()
model = transfer.get_model(fine_tune=False)

# Step 4: Train
Trainer.CHECKPOINT_PATH = 'artifacts/models/best_transfer_model.keras'
Trainer.EPOCHS = 10
trainer = Trainer(model)
history = trainer.train_with_dataset(train_ds, val_ds)

# Step 5: Evaluate -- resize test set separately (smaller, more manageable)
print("Resizing test set for evaluation...")
x_test_resized = tf.image.resize(x_test, (IMG_SIZE, IMG_SIZE)).numpy()

evaluator = Evaluator(model, x_test_resized, y_test)
results = evaluator.initiate_evaluation(history)

print(f"\nTransfer Learning Test Accuracy: {results['test_accuracy']:.4f}")

# Step 6: Log to tracker
tracker = ExperimentTracker()
tracker.log_experiment(
    experiment_name="transfer_learning_mobilenet",
    epochs=Trainer.EPOCHS,
    batch_size=BATCH_SIZE,
    augmentation=False,
    train_accuracy=history.history['accuracy'][-1],
    val_accuracy=history.history['val_accuracy'][-1],
    test_accuracy=results['test_accuracy']
)

print("Logged to experiment tracker.")