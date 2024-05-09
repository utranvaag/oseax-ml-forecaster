"""General helpfuncs for assisting in Optuna optimization"""

import warnings
import optuna
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback


class PyTorchLightningPruningCallback(Callback):
    """
    PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current
            evaluation of the objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned
            dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the
            names thus depend on how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple
        # times at epoch 0. The related page is:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


# Custom silent callback loss logger:
class LossLogger(Callback):
    """
    Custom loss logger. Automatically called at the end of each epoch.

    Note: The callback will give one more element in the loss_logger.val_loss
    as the model trainer performs a validationsanity check before the training
    begins.

    Example of use:\n
    loss_logger = LossLogger()

    model = SomeTorchForecastingModel(\n
        ...,\n
        nr_epochs_val_period=1,  # perform validation after every epoch\n
        pl_trainer_kwargs={"callbacks": [loss_logger]}\n
    )
    """

    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:

        self.train_loss.append(float(trainer.callback_metrics['train_loss']))

    def on_validation_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:

        val_loss = float(trainer.callback_metrics['val_loss'])
        self.val_loss.append(val_loss)


def print_callback(study, trial):
    """Prints info from optimization process"""

    print(
        f"[I Last MSE & prms : {trial.value}, {trial.params}"
    )
    print(
        f"[I Best MSE & prms : {study.best_value}, {study.best_trial.params}"
        "\n |----------------------------------"
    )


def print_callback_best(study, trial):
    """Prints info from optimization process - best trial & params only"""

    print(
        f"[I Best MSE & prms : {study.best_value}, {study.best_trial.params}"
    )


def logging_callback(study, frozen_trial):
    """Logging callback printing best params only when new best"""

    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )