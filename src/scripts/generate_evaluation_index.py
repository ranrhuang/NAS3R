from dataclasses import dataclass

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
# from pytorch_lightning import Trainer
from lightning import Trainer


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config, separate_dataset_cfg_wrappers
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfgWrapper
    from src.evaluation.evaluation_index_generator import (
        EvaluationIndexGenerator,
        EvaluationIndexGeneratorCfg,
    )
    from src.global_cfg import set_cfg


# python -m src.scripts.generate_evaluation_index  dataset/view_sampler@dataset.re10k.view_sampler=all 

@dataclass
class RootCfg:
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    index_generator: EvaluationIndexGeneratorCfg
    seed: int


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="generate_evaluation_index",
)
def train(cfg_dict: DictConfig):
    print("cfg_dict", cfg_dict)
    cfg = load_typed_config(cfg_dict, RootCfg,  {list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers})
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices="auto", strategy="auto")
    data_module = DataModule(cfg.dataset, cfg.data_loader, None)
    evaluation_index_generator = EvaluationIndexGenerator(cfg.index_generator)
    trainer.test(evaluation_index_generator, datamodule=data_module)
    evaluation_index_generator.save_index()


if __name__ == "__main__":
    train()