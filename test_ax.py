from torch.nn import *
from torchvision.datasets import CIFAR10
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import get_device
from nntoolbox.callbacks import *
from nntoolbox.metrics import Accuracy, Loss
from nntoolbox.losses import SmoothedCrossEntropy

from verta import Client
from verta.client import ExperimentRun
from experii.verta import ModelDBCB
from experii.ax import AxTuner

torch.backends.cudnn.benchmark=True
EXPERIMENT_NAME = "Hyperparameter Tuning"
client = Client("http://78351a12.ngrok.io")
proj = client.set_project("My second ModelDB project")


def model_fn(parameterization: Dict[str, Any]) -> nn.Module:
    class SEResNeXtShakeShake(ResNeXtBlock):
        def __init__(self, in_channels, reduction_ratio=16, cardinality=2, activation=nn.ReLU,
                     normalization=nn.BatchNorm2d):
            super(SEResNeXtShakeShake, self).__init__(
                branches=nn.ModuleList(
                    [
                        nn.Sequential(
                            ConvolutionalLayer(
                                in_channels, in_channels // 4, kernel_size=1, padding=0,
                                activation=activation, normalization=normalization
                            ),
                            ConvolutionalLayer(
                                in_channels // 4, in_channels, kernel_size=3, padding=1,
                                activation=activation, normalization=normalization
                            ),
                            SEBlock(in_channels, reduction_ratio)
                        ) for _ in range(cardinality)
                        ]
                ),
                use_shake_shake=True
            )

    class StandAloneMultiheadAttentionLayer(nn.Sequential):
        def __init__(
                self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=3,
                activation=nn.ReLU, normalization=nn.BatchNorm2d
        ):
            layers = [
                StandAloneMultiheadAttention(
                    num_heads=num_heads,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False
                ),
                activation(),
                normalization(num_features=out_channels),
            ]
            super(StandAloneMultiheadAttentionLayer, self).__init__(*layers)

    class SEResNeXtShakeShakeAttention(ResNeXtBlock):
        def __init__(self, num_heads, in_channels, reduction_ratio=16, cardinality=2, activation=nn.ReLU,
                     normalization=nn.BatchNorm2d):
            super(SEResNeXtShakeShakeAttention, self).__init__(
                branches=nn.ModuleList(
                    [
                        nn.Sequential(
                            ConvolutionalLayer(
                                in_channels=in_channels,
                                out_channels=in_channels // 2,
                                kernel_size=1,
                                activation=activation,
                                normalization=normalization
                            ),
                            StandAloneMultiheadAttentionLayer(
                                num_heads=num_heads,
                                in_channels=in_channels // 2,
                                out_channels=in_channels // 2,
                                kernel_size=3,
                                activation=activation,
                                normalization=normalization
                            ),
                            ConvolutionalLayer(
                                in_channels=in_channels // 2,
                                out_channels=in_channels,
                                kernel_size=1,
                                activation=activation,
                                normalization=normalization
                            ),
                            SEBlock(in_channels, reduction_ratio)
                        ) for _ in range(cardinality)
                        ]
                ),
                use_shake_shake=True
            )

    model = Sequential(
        ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU),
        SEResNeXtShakeShake(in_channels=16, activation=nn.ReLU),
        ConvolutionalLayer(
            in_channels=16, out_channels=32,
            activation=nn.ReLU,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=32),
        ConvolutionalLayer(
            in_channels=32, out_channels=64,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=64),
        ConvolutionalLayer(
            in_channels=64, out_channels=128,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=128),
        ConvolutionalLayer(
            in_channels=128, out_channels=256,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=256),
        ConvolutionalLayer(
            in_channels=256, out_channels=512,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=512),
        FeedforwardBlock(
            in_channels=512,
            out_features=10,
            pool_output_size=2,
            hidden_layer_sizes=(256, 128)
        )
    ).to(get_device())

    return model


def evaluate_fn(parameterization: Dict[str, Any], model: nn.Module, run: ExperimentRun) -> float:
    lr = parameterization["lr"]
    print("Evaluate at learning rate %f" % lr)

    # Set up train and validation data
    data = CIFAR10('data/', train=True, download=True, transform=ToTensor())
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
    train_dataset.dataset.transform = Compose(
        [
            RandomHorizontalFlip(),
            RandomResizedCrop(size=32, scale=(0.95, 1.0)),
            ToTensor()
        ]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    print("Number of batches per epoch " + str(len(train_loader)))


    # Set up model:

    # Setting up ModelDB:

    optimizer = SGD(model.parameters(), weight_decay=0.0001, lr=lr, momentum=0.9)
    learner = SupervisedImageLearner(
        train_data=train_loader,
        val_data=val_loader,
        model=model,
        criterion=SmoothedCrossEntropy().to(get_device()),
        optimizer=optimizer,
        mixup=True
    )

    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss()
    }

    callbacks = [
        ToDeviceCallback(),
        Tensorboard(),
        LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.024, T_max=405)),
        GradualLRWarmup(min_lr=0.024, max_lr=0.094, duration=810),
        LossLogger(),
        ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
        ModelDBCB(run=run, metrics=metrics),
    ]

    return learner.learn(
        n_epoch=1,
        callbacks=callbacks,
        metrics=metrics,
        final_metric='accuracy'
    )

tuner = AxTuner(
    client=client, evaluate_fn=evaluate_fn, model_fn=model_fn,
    specifications=[{"name": "lr", "type": "range", "bounds": [0.0, 1.0]}]
)
tuner.find_best_parameter()
