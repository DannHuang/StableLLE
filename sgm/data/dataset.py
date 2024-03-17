from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, get_worker_info
import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from functools import partial
from pytorch_lightning import LightningDataModule
from sgm.util import exists, instantiate_from_config

try:
    from sdata import create_dataset, create_dummy_dataset, create_loader
except ImportError as e:
    print("#" * 100)
    print("Datasets not yet available")
    print("to enable, we need to add stable-datasets as a submodule")
    print("please use ``git submodule update --init --recursive``")
    print("and do ``pip install -e stable-datasets/`` from the root of this repo")
    print("#" * 100)
    exit(1)


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_datapipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_datapipeline, **self.test_config.loader)


def worker_init_fn(_):
    worker_info = get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    # if isinstance(dataset, Txt2ImgIterableBaseDataset):
    #     split_size = dataset.num_records // worker_info.num_workers
    #     # reset num_records to the true number to retain reliable length information
    #     dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
    #     current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    #     return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    # else:
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class LDMDataModuleFromConfig(LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)

    def prepare_data(self):
        # for data_cfg in self.dataset_configs.values():
        #     instantiate_from_config_sr(data_cfg)
        pass

    def setup(self, stage=None):
        print('+++++++ Setting up Datasets +++++++')
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        # is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # if is_iterable_dataset or self.use_worker_init_fn:
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        # if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        # is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # if is_iterable_dataset or self.use_worker_init_fn:
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        # shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

