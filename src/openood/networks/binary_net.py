import torch.nn as nn
import torch


class BinaryNet(nn.Module):
    def __init__(self, backbone, hidden_layer_dim=256):
        super(BinaryNet, self).__init__()

        self.backbone = backbone
        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.proj_head = nn.Sequential(
            nn.Linear(feature_size * 2, hidden_layer_dim),
            nn.GELU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.GELU(),
            nn.Linear(hidden_layer_dim, 1),
        )

    def forward(self, x, return_feature=False, return_score=False):
        if return_feature:
            pred, feat = self.backbone(x, return_feature=True)
            return pred, feat
        elif return_score:
            feat1, feat2 = x
            score = self.proj_head(torch.cat([feat1, feat2], 1))
            return score
        else:
            return self.backbone(x, return_feature=False)
