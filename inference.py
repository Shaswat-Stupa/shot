import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.models import build_model
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

class Shot_Detector:

    def __init__(self) -> None:
        self.cfg = Config.fromfile("./slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py")
        self.checkpoint = "./best_top1_acc_epoch_213.pth"
        self.dummy_input = torch.randn(1, 1, 3, 16, 224, 224, device="cuda")
        # print(self.dummy_input)
        pass


    def turn_off_pretrained(self, cfg):
        # recursively find all pretrained in the model config,
        # and set them None to avoid redundant pretrain steps for testing
        if 'pretrained' in cfg:
            cfg.pretrained = None

        # recursively turn off pretrained value
        for sub_cfg in cfg.values():
            if isinstance(sub_cfg, dict):
                self.turn_off_pretrained(sub_cfg)

    def single_gpu_test(self, model):  # noqa: F811
            """Test model with a single gpu.

            This method tests model with a single gpu and
            displays test progress bar.

            Args:
                model (nn.Module): Model to be tested.
                data_loader (nn.Dataloader): Pytorch data loader.

            Returns:
                list: The prediction results.
            """
            model.eval()
            results = []
            
            with torch.no_grad():
                # result = model(return_loss=False, **self.dummy_input)
                result = model(self.dummy_input, return_loss=False)
            results.extend(result)

            return results

    def inference_pytorch(self):
        """Get predictions by pytorch models."""
        # print("\nTHIS FUNCTION IS GETTING CALLED\n")

        # remove redundant pretrain steps for testing
        self.turn_off_pretrained(self.cfg.model)

        # build the model and load self.checkpoint
        model = build_model(
            self.cfg.model, train_cfg=None, test_cfg=self.cfg.get('test_self.cfg'))

        # print(type(model))
        # print(model)

        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, self.checkpoint, map_location='cpu')


        model = build_dp(
            model, default_device, default_args=dict(device_ids=self.cfg.gpu_ids))
        print(type(model))
        # print(model)
        print("Single GPU function getting called...")
        outputs = self.single_gpu_test(model)
        return outputs

def main():
    detector = Shot_Detector()
    results = detector.inference_pytorch()
    print("Results: ", results)


if __name__ == "__main__":
    main()