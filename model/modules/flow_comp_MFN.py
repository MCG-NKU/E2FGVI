import torch
import torch.nn as nn
import torch.nn.functional as F

from mmflow.apis import init_model, inference_model


class MaskFlowNetS(nn.Module):
    """MaskFlowNetS network structure.
    The difference to the MaskFlowNetS is that
        1.
    Paper:
        MaskFlownet: Asymmetric Feature Matching With Learnable Occlusion Mask, CVPR, 2020
    Args:
        pretrained (str): path for pre-trained MaskFlowNetS. Default: None.
    """
    def __init__(
        self,
        use_pretrain=True,
        # pretrained='https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_slong_flyingchairs_384x448.pth',
        pretrained='./release_model/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.pth',
        config_file='../mmflow/configs/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.py',
        device='cuda:0',
        module_level=6
    ):
        super().__init__()

        self.maskflownetS = init_model(config_file, pretrained, device=device)

        if use_pretrain:
            if isinstance(pretrained, str):
                print("load pretrained MaskFlowNetS...")
                self.maskflownetS = init_model(config_file, pretrained, device=device)
                # load_checkpoint(self, pretrained, strict=True)
            elif pretrained is not None:
                raise TypeError('[pretrained] should be str or None, '
                                f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def centralize(img1, img2):
        """Centralize input images.
        Args:
            img1 (Tensor): The first input image.
            img2 (Tensor): The second input image.
        Returns:
            Tuple[Tensor, Tensor]: The first centralized image and the second
                centralized image.
        """
        rgb_mean = torch.cat((img1, img2), 2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 64.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        # ref = [(ref - self.mean) / self.std]
        # supp = [(supp - self.mean) / self.std]

        # ref, supp, _ = self.centralize(ref, supp)

        feat1, feat2 = self.maskflownetS.extract_feat(torch.cat((ref, supp), dim=1))
        flows_stage1, mask_stage1 = self.maskflownetS.decoder(
            feat1, feat2, return_mask=True)

        return flows_stage1

    def forward(self, ref, supp):
        """Forward function of MaskFlowNetS.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # # upsize to a multiple of 32
        # h, w = ref.shape[2:4]
        # w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        # h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        # ref = F.interpolate(input=ref,
        #                     size=(h_up, w_up),
        #                     mode='bilinear',
        #                     align_corners=False)
        # supp = F.interpolate(input=supp,
        #                      size=(h_up, w_up),
        #                      mode='bilinear',
        #                      align_corners=False)

        # upsize to a multiple of 64
        h, w = ref.shape[2:4]
        w_up = w if (w % 64) == 0 else 64 * (w // 64 + 1)
        h_up = h if (h % 64) == 0 else 64 * (h // 64 + 1)
        ref = F.interpolate(input=ref,
                            size=(h_up, w_up),
                            mode='bilinear',
                            align_corners=False)
        supp = F.interpolate(input=supp,
                             size=(h_up, w_up),
                             mode='bilinear',
                             align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(input=self.compute_flow(ref, supp)['level2'],
                             size=(h, w),
                             mode='bilinear',
                             align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


def test_MFN():
    # Specify the path to model config and checkpoint file
    config_file = '../../../mmflow/configs/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.py'
    checkpoint_file = '../../release_model/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.pth'

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    pass


if __name__ == "__main__":
    test_MFN()
