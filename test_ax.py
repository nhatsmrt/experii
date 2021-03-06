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

# Set up ModelDB:
client = Client(CLIENT_PARA) # supply your own ModelDB'S client parameters here (see VertaAI's notebooks)
proj = client.set_project("My second ModelDB project")
exp = client.set_experiment(EXPERIMENT_NAME)

# Define model generating function:
def model_fn(parameterization: Dict[str, Any]) -> nn.Module:
    model = Sequential(
        ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU),
        ResidualBlockPreActivation(in_channels=16, activation=nn.ReLU),
        ConvolutionalLayer(in_channels=16, out_channels=32, kernel_size=3, activation=nn.ReLU),
        ResidualBlockPreActivation(in_channels=32, activation=nn.ReLU),
        FeedforwardBlock(
            in_channels=32,
            out_features=10,
            pool_output_size=2,
            hidden_layer_sizes=(64, 32)
        )
    ).to(get_device())

    return model

# Define evaluating function:
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
        LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.024, T_max=405)),
        LossLogger(),
        ModelDBCB(run=run, filepath="weights/model.pt", metrics=metrics, monitor='accuracy', mode='max')
    ]

    return learner.learn(
        n_epoch=20,
        callbacks=callbacks,
        metrics=metrics,
        final_metric='accuracy'
    )

tuner = AxTuner(
    client=client, evaluate_fn=evaluate_fn, model_fn=model_fn,
    specifications=[{"name": "lr", "type": "range", "bounds": [0.0, 1.0]}]
)
tuner.find_best_parameter()
