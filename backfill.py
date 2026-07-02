from src.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

tracker.log_experiment(
    experiment_name="baseline_no_augmentation",
    epochs=30,
    batch_size=64,
    augmentation=False,
    train_accuracy=0.5280,
    val_accuracy=0.4102,
    test_accuracy=0.4234
)

tracker.log_experiment(
    experiment_name="with_augmentation",
    epochs=30,
    batch_size=64,
    augmentation=True,
    train_accuracy=0.0,
    val_accuracy=0.0,
    test_accuracy=0.4567
)

print("Backfill done.")