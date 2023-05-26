import logging
from omegaconf import DictConfig, OmegaConf
import hydra

import ipdb

from .dataset import load_fc_data, init_stratified_dataloader
from .models import GCN
from .components import optimizers_factory, lr_scheduler_factory



@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    final_pearson, labels, site = load_fc_data(cfg)

    dataloaders = init_stratified_dataloader(cfg, final_pearson, labels, site)

    ## test GCN model
    model = GCN(cfg)
    for batch in dataloaders[0]:
        print(batch.x.shape)
        print(batch.edge_index.shape)
        print(batch.y.shape)
        output = model(batch.x, batch.edge_index)
        print(output.shape)
        # ipdb.set_trace()
        break


    # # test logger
    # logger = logging.getLogger()
    # logger.info("testing this logger")

    # # `cfg` is your DictConfig object
    # cfg_str = OmegaConf.to_yaml(cfg)
    # logger.info("Configuration:\n%s", cfg_str)

    # optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    # lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    
    # print(optimizers)
    # print(lr_schedulers)
    # optimizer = optimizers[0]
    # lr_scheduler = lr_schedulers[0]

    # # Now let's simulate some training steps to test the learning rate scheduler
    # for step in range(1, cfg.total_steps + 1):
    #     # simulate an optimizer step
    #     optimizer.step()

    #     # update learning rate using the scheduler
    #     lr_scheduler.update(optimizer=optimizer, step=step)

    #     # check if the learning rate is updated correctly
    #     for param_group in optimizer.param_groups:
    #         print(f'Step: {step}, Learning Rate: {param_group["lr"]}')



if __name__ == "__main__":
    main()
    ipdb.set_trace()
