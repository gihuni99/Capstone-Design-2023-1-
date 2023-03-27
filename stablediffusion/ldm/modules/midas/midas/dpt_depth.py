import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False, # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


#위 코드는 DPTDepthModel이라는 클래스를 정의하는 코드입니다. DPTDepthModel은 DPT 클래스를 상속받아서 만들짐
#DPT는 Deep Hierarchical Multi-Scale Patch-based Network for Depth Estimation (DPT) 모델을 구현한 클래스
#DPT는 위에 정의되어 있고, 주석 설명을 달아 놓았다.
class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        #head는 DPT에서 output을 만드는 최종 단계에서 사용되고,
        #dep2img에서 사용되는 Depth estimation인 DPTDepthModel에서 사용되기 위해 따로 선언된 것이다.
        head = nn.Sequential( #nn.Sequential은 PyTorch에서 여러 개의 레이어(layer)를 순차적으로 연결하여 
                              #하나의 신경망 모델을 생성할 때 사용하는 클래스

            #입력 feature map에서 feature를 추출하고, feature map의 크기를 절반으로 줄입니다.
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),

            #feature map의 크기를 2배로 늘려줍니다. 이 때, bilinear 보간법을 사용하여 크기를 조절합니다. 
            #align_corners는 보간 과정에서 각 픽셀의 좌표를 정확히 일치시키는 옵션입니다.
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),

            #feature map에서 feature를 추출하고, feature map의 크기를 유지합니다.
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),

            #비선형성인 ReLU 함수를 적용합니다. 
            #True 옵션은 inplace 연산을 수행하는 것을 의미합니다. (입력과 출력의 메모리를 공유하여 메모리를 절약할 수 있음)
            nn.ReLU(True),

            #feature map에서 depth map을 예측하기 위한 필터를 적용합니다.
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),

            #ReLU 함수를 적용하여 양수로 만듭니다. 
            #non_negative=True인 경우에만 ReLU 함수를 사용하며, 그렇지 않은 경우에는 아무 동작도 하지 않는 Identity 레이어를 사용합니다.
            nn.ReLU(True) if non_negative else nn.Identity(),

            #출력값을 일정하게 유지하기 위한 함수, 입력값을 그대로 출력값으로 가져온다.
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
           self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)

