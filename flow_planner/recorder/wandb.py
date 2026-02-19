import wandb
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Dict, Optional, Union
from flow_planner.recorder import RecorderBase


class WandbRecorder(RecorderBase):
    def __init__(
        self,
        project: str,
        name: str,
        config: Union[Dict, DictConfig],
        wandb_id: Optional[str] = None,
        resume: str = "allow",
        mode: str = "online",
        **kwargs,
    ):
        super().__init__()

        run_id = wandb_id if wandb_id is not None else kwargs.get("id", None)
        if "id" in kwargs:
            del kwargs["id"]

        if isinstance(config, (DictConfig, ListConfig)):
            config = OmegaConf.to_container(config, resolve=True)

        # 【核心修改】添加 settings=wandb.Settings(start_method="thread")
        # 这有助于防止 DDP 进程退出时的信号死锁
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            id=run_id,
            resume=resume,
            mode=mode,
            settings=wandb.Settings(start_method="thread"),
            **kwargs,
        )
        self.id = self.run.id

    def record_loss(self, loss: Dict, step: int):
        wandb.log(loss, step=step)

    def record_metric(self, metrics: Dict, step: int):
        wandb.log(metrics, step=step)

    def close(self):
        if self.run is not None:
            # quiet=True 可以减少打印，但不能完全解决卡死
            # 配合 start_method="thread" 效果最好
            try:
                self.run.finish(quiet=True)
            except Exception:
                pass
