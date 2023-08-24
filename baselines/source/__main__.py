import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

import ipdb

from .dataset import dataset_factory
from .models import model_factory
from .components import optimizers_factory, lr_scheduler_factory
from .training import training_factory

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'




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

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_node:{cfg.dataset.node}"

    for _ in range(cfg.repeat_time):

        if cfg.is_wandb:
            run = wandb.init(project=cfg.project, reinit=True,
                             group=f"{group_name}", tags=[f"{cfg.dataset.name}, {cfg.model.name}"])
        logging.info(OmegaConf.to_yaml(cfg)) 
        model_training(cfg)

        if cfg.is_wandb:
            wandb.finish()



def sweep_agent_manager():
    wandb.init()
    sweep_config = wandb.config
    sweep_config = OmegaConf.create(sweep_config.as_dict())

    # Merging sweep config with config
    cfg = OmegaConf.load("source/conf/config.yaml")
    ipdb.set_trace()

    cfg.defaults.model.num_layers = sweep_config.model.num_layers
    cfg.defaults.model.hidden_dim = sweep_config.model.hidden_dim
    cfg.defaults.model.K_neigs = sweep_config.model.K_neigs

    # cfg.defaults.dataset.perc_edges = sweep_config.dataset.perc_edges

    print("#############################################")
    print(cfg)
    main(cfg)
    

if __name__ == "__main__":
    # cfg = OmegaConf.load("source/conf/config.yaml")
    # if cfg.doing_sweep:
    #     wandb.agent(sweep_id=cfg.sweep_id, function=sweep_agent_manager)
    # else:
    #     main()
    main()
