import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F
import numpy as np
import einops


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BatchNorm1d_P(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  input: (b, n, c)
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm2d_P(nn.BatchNorm2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: (b, n, k, c)
        x = x.permute(0, 3, 1, 2)   # (b, c, n, k)
        x = super().forward(x)
        return x.permute(0, 2, 3, 1)    # (b, n, k, c)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1).type(torch.int64)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes, nsample):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.attn_planes = attn_planes = out_planes // share_planes
        self.k = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.fc_delta = nn.Sequential(nn.Linear(3, 3),
                                      BatchNorm2d_P(3),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))

        self.fc_gamma = nn.Sequential(BatchNorm2d_P(mid_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, attn_planes),
                                      BatchNorm2d_P(attn_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(attn_planes, attn_planes))

    def forward(self, xyz, features):
        """
        Input:
            xyz: (b, n, 3)
            features: (b, n, in_planes)
        Output:
            res: (b, n, out_planes)
        """
        # find k nearest neighbors for each point
        knn_idx = pointops.knnquery(self.k, xyz)    # (b, n, k)
        if xyz.size(1) < self.k:
            knn_idx = knn_idx[:, :, :xyz.size(1)]  # avoid repetition of 0th point

        knn_xyz = index_points(xyz, knn_idx)    # (b, n, k, 3)

        q = self.linear_q(features) # (b, n, mid_planes)
        k = index_points(self.linear_k(features), knn_idx)  # (b, n, k, mid_planes)
        v = index_points(self.linear_v(features), knn_idx)  # (b, n, k, out_planes)

        # position encoding
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # (b, n, k, out_planes)

        # attention weights
        attn = self.fc_gamma(q[:, :, None] - k +
                             einops.reduce(pos_enc, "b n k (i j) -> b n k j", reduction="sum", j=self.mid_planes))   # (b, n, k, attn_planes)
        attn = F.softmax(attn, dim=-2)

        # product
        res = torch.einsum("b n k s a, b n k a -> b n s a",
                           einops.rearrange(v + pos_enc, "b n k (s a) -> b n k s a", s=self.share_planes),
                           attn)    # (b, n, share_planes, attn_planes)
        res = einops.rearrange(res, "b n s a -> b n (s a)") # (b, n, out_planes)

        return res


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16, embed_dim=64):
        super(Bottleneck, self).__init__()

        self.linear1 = nn.Linear(in_planes+embed_dim, planes+embed_dim, bias=False)
        self.bn1 = BatchNorm1d_P(planes+embed_dim)

        self.transformer = PointTransformerLayer(planes+embed_dim, planes+embed_dim, share_planes, nsample)
        self.bn2 = BatchNorm1d_P(planes+embed_dim)

        self.linear3 = nn.Linear(planes+embed_dim, planes, bias=False)
        self.bn3 = BatchNorm1d_P(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        """
        Input:
            inputs: [xyz, features, temb]
        Output:
            [xyz, new_features]
        """
        xyz, features, temb = inputs  # (b,n,3), (b,n,c), (b,n,64)
        identity = features
        features = self.relu(self.bn1(self.linear1(torch.cat([features, temb], dim=2))))    # (b,n,c+64)
        features = self.relu(self.bn2(self.transformer(xyz, features)))    # (b,n,c+64)
        features = self.bn3(self.linear3(features))    # (b,n,c)
        features += identity
        features = self.relu(features)
        return [xyz, features, temb]  # (b,n,3), (b,n,c), (b,n,64)


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride, nneighbor=16):
        super().__init__()
        self.stride, self.nneighbor = stride, nneighbor
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.bn = BatchNorm2d_P(out_planes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.bn = BatchNorm1d_P(out_planes)
            self.swish = Swish()

    def forward(self, inputs):
        """
        Input:
            px: [xyz, features, temb]
        Output:
            [xyz, new_features, temb] if stride == 1 else [new_xyz, new_features, temb]
        """
        xyz, features, temb = inputs  # (b,n,3), (b,n,c), (b,n,64)
        new_xyz, new_features = xyz, features
        if self.stride != 1:
            # 1. farthest point sampling
            npoint = xyz.size(1) // self.stride
            fps_idx = pointops.furthestsampling(xyz, npoint)    # (b,m)
            new_xyz = index_points(xyz, fps_idx)    # (b,m,3)
            # 2. knn
            knn_idx = pointops.knnquery(self.nneighbor, xyz, new_xyz)   # (b,m,k)
            grouped_xyz = index_points(xyz, knn_idx)    # (b,m,k,3)
            grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]    # (b,m,k,3)
            grouped_features = torch.cat([grouped_xyz_norm, index_points(features, knn_idx)], dim=-1)   # (b,m,k,3+in_planes)
            # 3. mlp
            grouped_features = self.relu(self.bn(self.linear(grouped_features)))    # (b,m,k,out_planes)
            # 4. local max pooling
            new_features = torch.max(grouped_features, dim=2)[0]    # (b,m,out_planes)
        else:
            # new_features = self.relu(self.bn(self.linear(features)))  # (b,n,out_planes)
            new_features = self.swish(self.bn(self.linear(features)))    # (b,n,out_planes)
        return [new_xyz, new_features, temb[:, :new_features.size(1), :]]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None, global_pooling=None):
        super().__init__()
        if out_planes is None:
            self.global_pooling = global_pooling
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes),
                                         BatchNorm1d_P(in_planes),
                                         nn.ReLU(inplace=True))
            # applied to global feature
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes),
                                         nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes),
                                         BatchNorm1d_P(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes),
                                         BatchNorm1d_P(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, px1, px2=None):
        if px2 is None:
            p1, x1 = px1    # (b,n,3) and (b,n,c) from the main branch

            if self.global_pooling == 'avg':
                x_mean = self.linear2(x1.mean(dim=1, keepdim=True)).repeat(1, x1.size(1), 1)    # (b,n,c)
                x = torch.cat([x1, x_mean], dim=2)  # (b,n,c*2)
            elif self.global_pooling == 'max':
                x_max = self.linear2(x1.max(dim=1, keepdim=True)[0]).repeat(1, x1.size(1), 1)    # (b,n,c)
                x = torch.cat([x1, x_max], dim=2)  # (b,n,c*2)
            else:
                raise Exception('global_pooling must be set to either avg or max...')

            x = self.linear1(x)  # (b,n,c)
        else:
            p1, x1 = px1    # (b,n,3) and (b,n,c) from skip connections.
            p2, x2 = px2    # (b,n/4,3) and (b,n/4,c*2) from the main branch.

            # apply interpolation to the main branch
            dist, idx = pointops.nearestneighbor(p1, p2)    # find 3 nearest neighbors of p1 in p2
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_x = pointops.interpolation(self.linear2(x2).permute(0, 2, 1), idx, weight).permute(0, 2, 1)

            # summation
            x = interpolated_x + self.linear1(x1)   # (b,n,c)
        return x


if __name__ == '__main__':
    # in_planes, out_planes, stride = 3, 32, 1
    # transition_down_block = TransitionDown(in_planes, out_planes, stride).cuda()
    # transition_down_block([xyz, xyz])

    # in_planes, out_planes, stride, nneighbor = 32, 64, 4, 16
    # transition_down_block = TransitionDown(in_planes, out_planes, stride, nneighbor).cuda()
    # transition_down_block([xyz, features])

    px1 = [torch.randn(16, 16, 3).cuda(), torch.randn(16, 16, 256).cuda()]
    px2 = [torch.randn(16, 4, 3).cuda(), torch.randn(16, 4, 512).cuda()]
    in_planes, out_planes = 512, 256
    transition_up_block = TransitionUp(in_planes, out_planes).cuda()
    transition_up_block(px1, px2)

    px = [torch.randn(16, 4, 3).cuda(), torch.randn(16, 4, 512).cuda()]
    in_planes = 512
    transition_up_block = TransitionUp(in_planes).cuda()
    transition_up_block(px)
