from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias
import logging


@dataclass(frozen=True)
class StateActionReward:
    state: Any
    action: Any
    reward: float


Trajectory: TypeAlias = list[StateActionReward]
DataSet: TypeAlias = list[Trajectory]


class TrainFlow(ABC):
    init_k: int = 0
    k: int = 0

    def create_dataset(self) -> DataSet:
        return []

    @abstractmethod
    def update_dataset(self, D: DataSet):
        pass

    @abstractmethod
    def train_step(self, D: DataSet):
        pass

    def eval_step(self):
        pass

    def save_model(self, k: int):
        pass

    def run(self, *, max_k: int, eval_interval: int, save_interval: int):
        logging.info("start training ...")
        D = self.create_dataset()

        for k in range(self.init_k, max_k):
            self.k = k

            logging.debug(f"k={k}, update dataset ...")
            self.update_dataset(D)

            logging.debug(f"k={k}, training ..., #dataset={len(D)}")
            self.train_step(D)

            if (eval_interval != 0) and ((k + 1) % eval_interval == 0):
                logging.debug(f"k={k}, evaling ...")
                self.eval_step()

            if (save_interval != 0) and ((k + 1) % save_interval == 0):
                logging.debug(f"k={k}, saving model ...")
                self.save_model(k)
