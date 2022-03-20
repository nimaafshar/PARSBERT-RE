import abc
import pathlib

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Optional, MutableMapping, Union, List, Dict, Tuple
from dataclasses import dataclass

from ..settings import Config


@dataclass
class TrainingArguments:
    epochs: int = 10
    clip: float = 0.0
    train_callback_interval: int = 100


class Trainer(abc.ABC):
    def __init__(self, model: nn.Module,
                 device: torch.device,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 arguments: TrainingArguments,
                 save_model: bool = True,
                 output_path: pathlib.Path = Config.OUTPUT_PATH):
        self._model: torch.nn.Module = model
        self._save_model: bool = save_model
        self._output_path: pathlib.Path = output_path

        self._device: torch.device = device
        self._loss_function: nn.Module = loss_function
        self._optimizer: torch.optim.Optimizer = optimizer
        self._scheduler: nn.Module = scheduler

        self._arguments: TrainingArguments = arguments

        self._training_data_loader: torch.utils.data.DataLoader = self.get_training_data_loader()
        self._validation_data_loader: torch.utils.data.DataLoader = self.get_validation_data_loader()
        self._test_data_loader: torch.utils.data.DataLoader = self.get_test_data_loader()

        self._valid_loss_min: float = np.inf

    @property
    def total_steps(self) -> int:
        return self._arguments.epochs * len(self._training_data_loader)

    @abc.abstractmethod
    def get_training_data_loader(self) -> torch.utils.data.DataLoader:
        pass

    @abc.abstractmethod
    def get_validation_data_loader(self) -> torch.utils.data.DataLoader:
        pass

    @abc.abstractmethod
    def get_test_data_loader(self) -> torch.utils.data.DataLoader:
        pass

    def _move_to_device(self, value: Union[MutableMapping[str, torch.Tensor], torch.Tensor]):
        if isinstance(value, torch.Tensor):
            return value.to(self._device)
        elif isinstance(value, MutableMapping):
            for key in value.keys():
                value[key] = value[key].to(self._device)
            return value
        else:
            raise TypeError(f"{type(value)} is not supported for moving to device.")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, losses: Optional[List[float]] = None) -> \
            Dict[str, float]:
        # override this function if you want to use different metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        if losses:
            metrics['loss'] = np.mean(losses)
        return metrics

    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        statement = "["
        for key, value in metrics.items():
            statement += f"{key}:{value:.4f},"
        statement += "]"
        print(statement)

    def _train_callback(self, step: int, total_steps: int, y_true: np.ndarray, y_pred: np.ndarray, losses: List[float]):
        print(f"step {step}/{total_steps}:")
        metrics: Dict[str, float] = self._compute_metrics(y_true, y_pred, losses)
        self._print_metrics(metrics)

    def _train_operation(self) -> Tuple[float, Dict[str, float]]:
        losses = []
        y_pred = np.empty()
        y_true = np.empty()
        # put model in training mode
        self._model.train()
        step = 0

        for data in tqdm(self._training_data_loader, total=len(self._training_data_loader), desc="Training... "):
            step += 1
            data: dict

            # move tensors to GPU if CUDA is available
            data = self._move_to_device(data)
            targets = data.pop('targets')

            # clear the gradients of all optimized variables
            self._optimizer.zero_grad()

            # compute predicted outputs by passing inputs to the model
            # data is a Mapping containing input_ids,token_type_ids,attention_mask
            outputs = self._model(**data)

            # convert output probabilities to predicted class
            _, predictions = torch.max(outputs, dim=1)

            # calculate the batch loss
            loss = self._loss_function(outputs, targets)

            # accumulate all the losses
            losses.append(loss.item())

            # compute gradient of the loss with respect to model parameters
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if self._arguments.clip > 0.0:
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=self._arguments.clip)

            # perform optimization step
            self._optimizer.step()

            # perform scheduler step
            self._scheduler.step()

            # adding new predictions
            y_pred = np.append(y_pred, predictions.cpu().detach().numpy())
            y_true = np.append(y_true, targets.cpu().detach().numpy())

            # print metric
            if step % self._arguments.train_callback_interval == 0:
                self._train_callback(step, len(self._training_data_loader), y_true, y_pred, losses)

        return np.mean(losses), self._compute_metrics(y_true, y_pred, losses)

    def _validation_operation(self) -> Tuple[float, Dict[str, float]]:
        # put model into evaluation mode
        self._model.eval()

        losses = []
        y_pred = np.empty()
        y_true = np.empty()
        step = 0
        with torch.no_grad():
            for data in tqdm(self._validation_data_loader, total=len(self._validation_data_loader),
                             desc="Validation... "):
                step += 1
                data: dict

                # move tensors to GPU if CUDA is available
                data = self._move_to_device(data)
                targets = data.pop('targets')

                # compute predicted outputs by passing inputs to the model
                outputs = self._model(**data)

                # convert output probabilities to predicted class
                _, predictions = torch.max(outputs, dim=1)

                # calculate the batch loss
                loss = self._loss_function(outputs, targets)

                # accumulate all the losses
                losses.append(loss.item())

                # adding new predictions
                y_pred = np.append(y_pred, predictions.cpu().detach().numpy())
                y_true = np.append(y_true, targets.cpu().detach().numpy())

        return np.mean(losses), self._compute_metrics(y_true, y_pred, losses)

    def _save(self) -> None:
        torch.save(self._model, self._output_path)

    def train(self) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        train_history = []
        valid_history = []
        for epoch in tqdm(range(1, epochs + 1), desc="Epochs... "):
            print(f"epoch {epoch}/{epochs}")
            train_loss, train_metrics = self._train_operation()
            print("train:", end=" ")
            self._print_metrics(train_metrics)

            valid_loss, valid_metrics = self._validation_operation()
            print("validation:", end=" ")
            self._print_metrics(valid_metrics)

            if valid_loss < self._valid_loss_min:
                print(f"Validation loss decreased {self._valid_loss_min:.6f} -> {valid_loss:.6f}: saving model...")
                self._valid_loss_min = valid_loss
                self._save()

            train_history.append(train_metrics)
            valid_history.append(valid_metrics)
        return train_history, valid_history

    def predict_tests(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        data_loader = self._test_data_loader

        y_pred_batches = []
        y_pred_probs_batches = []
        y_true_batches = []

        self._model.eval()
        with torch.no_grad():
            for data in tqdm(data_loader, total=len(data_loader), desc="Predicting tests"):
                # move tensors to GPU if CUDA is available
                data = self._move_to_device(data)
                targets = data.pop('targets')

                # compute predicted outputs by passing inputs to the model
                outputs = self._model(**data)

                # convert output probabilities to predicted class
                _, predictions = torch.max(outputs, dim=1)

                y_pred_batches.extend(predictions)
                y_pred_probs_batches.extend(F.softmax(outputs, dim=1))
                y_true_batches.extend(targets)

        y_pred = torch.stack(y_pred_batches).cpu().detach().numpy()
        y_pred_probs = torch.stack(y_pred_probs_batches).cpu().detach().numpy()
        y_true = torch.stack(y_true_batches).cpu().detach().numpy()

        return y_pred, y_pred_probs, y_true, self._compute_metrics(y_true, y_pred)
