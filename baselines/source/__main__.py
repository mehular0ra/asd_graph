from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore

import ipdb

from .dataset.fc_small import load_fc_data
from .dataset.dataloader import init_stratified_dataloader

from collections import Counter


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    final_pearson, labels, site = load_fc_data(cfg)

    dataloaders = init_stratified_dataloader(cfg, final_pearson, labels, site)
    for dl in dataloaders:
        for batch in dl:
            print(batch)  # or inspect batch in any other way you want
    print()
    train_sites = [data.site for data in dataloaders[0].dataset]
    test_sites = [data.site for data in dataloaders[1].dataset]
    print("Training dataset site breakdown:", Counter(train_sites))
    print("Testing dataset site breakdown:", Counter(test_sites))



if __name__ == "__main__":
    main()
    ipdb.set_trace()
