import click
import keras
import keras.applications.resnet50
import numpy
import tensorflow

import keras_resnet.models

_benchmarks = {
    "CIFAR-10": keras.datasets.cifar10,
    "CIFAR-100": keras.datasets.cifar100,
    "MNIST": keras.datasets.mnist
}


_models = {
    "ResNet-18": keras_resnet.models.ResNet18,
    "ResNet-34": keras_resnet.models.ResNet34,
    "ResNet-50": keras_resnet.models.ResNet50,
    "ResNet-101": keras_resnet.models.ResNet101,
    "ResNet-152": keras_resnet.models.ResNet152,
    "ResNet-200": keras_resnet.models.ResNet200
}


@click.command()
@click.option(
    "--benchmark",
    default="CIFAR-10",
    type=click.Choice(
        [
            "CIFAR-10",
            "CIFAR-100",
            "MNIST"
        ]
    )
)
@click.option("--device", default=0)
@click.option(
    "--name",
    default="ResNet-50",
    type=click.Choice(
        [
            "ResNet-18",
            "ResNet-34",
            "ResNet-50",
            "ResNet-101",
            "ResNet-152",
            "ResNet-200"
        ]
    )
)
def __main__(benchmark, device, name):
    configuration = tensorflow.ConfigProto()

    configuration.gpu_options.allow_growth = True

    configuration.gpu_options.visible_device_list = str(device)

    session = tensorflow.Session(config=configuration)

    keras.backend.set_session(session)

    (training_x, training_y), _ = _benchmarks[benchmark].load_data()

    training_x = training_x.astype(numpy.float16)

    training_x = keras.applications.resnet50.preprocess_input(training_x)

    training_y = keras.utils.np_utils.to_categorical(training_y)

    shape, classes = training_x.shape[1:], training_y.shape[-1]

    x = keras.layers.Input(shape)

    x = _models[name](x)

    y = keras.layers.Flatten()(x.output)

    y = keras.layers.Dense(classes, activation="softmax")(y)

    model = keras.models.Model(x.input, y)

    optimizer = keras.optimizers.Adam()

    loss = "categorical_crossentropy"

    metrics = [
        "accuracy"
    ]

    model.compile(optimizer, loss, metrics)

    pathname = "{}.hdf5".format(name)

    model_checkpoint = keras.callbacks.ModelCheckpoint(pathname)

    pathname = "{}.csv".format(name)

    csv_logger = keras.callbacks.CSVLogger(pathname)

    callbacks = [
        csv_logger,
        model_checkpoint
    ]

    model.fit(
        batch_size=256,
        callbacks=callbacks,
        epochs=200,
        validation_split=0.1,
        x=training_x,
        y=training_y
    )

if __name__ == "__main__":
    __main__()
