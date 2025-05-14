import torch.nn as nn


class ContrastiveNet(nn.Module):
    def __init__(self, backbone, proj_type, hidden_layer_dim=256, feat_dim=128):
        super(ContrastiveNet, self).__init__()

        self.backbone = backbone
        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        if proj_type == "mlp":
            self.proj_head = nn.Sequential(
                nn.Linear(feature_size, hidden_layer_dim),
                nn.GELU(),
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.GELU(),
                nn.Linear(hidden_layer_dim, feat_dim),
            )
        elif proj_type == "identity":
            self.proj_head = nn.Identity()

    def forward(self, x, return_feat_pre=False, return_feat_post=False):
        pred, feat_pre = self.backbone(x, return_feature=True)
        feat_post = self.proj_head(feat_pre)

        if return_feat_pre and return_feat_post:
            return pred, feat_pre, feat_post
        elif return_feat_pre:
            return pred, feat_pre
        elif return_feat_post:
            return pred, feat_post
        else:
            return pred
