from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore

import ipdb

from .dataset.fc_small import load_fc_data
from .dataset.dataloader import init_stratified_dataloader
from .models.GCN import GCN

from collections import Counter


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



if __name__ == "__main__":
    main()
    ipdb.set_trace()
