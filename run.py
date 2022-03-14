import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
import copy
import pytorch_lightning as pl

from model.config import ex
from model.modules import PerceiverVL
from model.datamodules.multitask_datamodule import MTDataModule
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = PerceiverVL(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}_{_config["model_type"]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["gpus"]
        if isinstance(_config["gpus"], int)
        else len(_config["gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    trainer = pl.Trainer(
        gpus=_config["gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        strategy="ddp",
        benchmark=True,
        deterministic=False,
        auto_lr_find=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        multiple_trainloader_mode="max_size_cycle",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
