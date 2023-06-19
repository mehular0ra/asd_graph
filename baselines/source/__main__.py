import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

import ipdb

from .dataset import dataset_factory
from .models import model_factory
from .components import optimizers_factory, lr_scheduler_factory
from .training import training_factory



def model_training(cfg: DictConfig):

    dataloaders = dataset_factory(cfg)
    model = model_factory(cfg)
    print(model)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    training = training_factory(cfg=cfg,
                                model=model,
                                optimizers=optimizers,
                                lr_schedulers=lr_schedulers,
                                dataloaders=dataloaders)
    training.train()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    group_name = f"{cfg.dataset.name}_{cfg.model.name}"

    for _ in range(cfg.repeat_time):

        if cfg.is_wandb:
            run = wandb.init(project=cfg.project, reinit=True,
                             group=f"{group_name}", tags=[f"{cfg.dataset.name}, {cfg.model.name}"])
            # save config
            wandb.init(project=cfg.project, config=cfg)
            # wandb.config.update(cfg)

        logging.info(OmegaConf.to_yaml(cfg)) 
        model_training(cfg)

        if cfg.is_wandb:
            wandb.finish()




if __name__ == "__main__":
    main()
    ipdb.set_trace()
