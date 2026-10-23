# Assignment 3 - Part C
# Tracking MNIST experiments with MLflow and DagsHub

import dagshub
import mlflow
import mlflow.tensorflow
from tensorflow import keras

# connect to the DagsHub repo (this also sets the MLflow tracking URI)
dagshub.init(repo_owner="AhmedWalidBarakat", repo_name="mnist-mlflow-assignment3", mlflow=True)
print("Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("mnist-dense-nn")

# autolog records loss/accuracy for every epoch + the trained model
mlflow.tensorflow.autolog()

# same random seed every time so runs are comparable
keras.utils.set_random_seed(42)

# load MNIST and scale pixels from 0-255 down to 0-1
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


def build_model(hidden_layers, nodes, optimizer):
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())

    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(nodes, activation="relu"))

    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_experiment(run_name, hidden_layers, nodes, optimizer, batch_size, epochs):
    with mlflow.start_run(run_name=run_name):
        # the hyperparameters the assignment asks for
        mlflow.log_params({
            "hidden_layers": hidden_layers,
            "nodes_per_layer": nodes,
            "optimizer": optimizer,
            "batch_size": batch_size,
        })

        model = build_model(hidden_layers, nodes, optimizer)

        # 10% of the training data is held out for validation
        model.fit(x_train, y_train,
                  validation_split=0.1,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2)

        # final check on the test set that the model never saw
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"{run_name}: test accuracy = {test_acc:.4f}")


# baseline + 3 more experiments
experiments = [
    # name                          layers  nodes  optimizer  batch  epochs
    ("baseline_1x128_adam_bs64_e50",  1,     128,   "adam",    64,    50),
    ("deeper_2x256_adam_bs64_e50",    2,     256,   "adam",    64,    50),
    ("sgd_1x128_bs64_e50",            1,     128,   "sgd",     64,    50),
    ("bigbatch_2x128_adam_bs128_e20", 2,     128,   "adam",    128,   20),
]

for name, layers, nodes, opt, batch, epochs in experiments:
    run_experiment(name, layers, nodes, opt, batch, epochs)
