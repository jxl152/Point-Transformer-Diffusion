import torch
import torch.nn as nn
import numpy as np

from model.point_transformer.point_transformer_blocks import Bottleneck, TransitionDown, TransitionUp, BatchNorm1d_P, Swish


# borrow from https://github.com/jxl152/Point-Transformer, and refer to https://github.com/Pointcept/Pointcept
class PointTransformerSeg(nn.Module):

    def __init__(self, block, enc_blocks, dec_blocks, in_channels, num_classes, embed_dim, stride, nsample, global_pooling):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes, share_planes = in_channels, [32, 64, 128, 256, 512], 8
        self.embed_dim = embed_dim
        self.global_pooling = global_pooling

        self.enc1 = self._make_enc(block, planes[0], enc_blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(block, planes[1], enc_blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(block, planes[2], enc_blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(block, planes[3], enc_blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(block, planes[4], enc_blocks[4], share_planes, stride[4], nsample[4])

        self.dec5 = self._make_dec(block, planes[4], dec_blocks[4], share_planes, nsample[4], is_head=True)   # transform p5
        self.dec4 = self._make_dec(block, planes[3], dec_blocks[3], share_planes, nsample[3]) # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], dec_blocks[2], share_planes, nsample[2]) # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], dec_blocks[1], share_planes, nsample[1]) # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], dec_blocks[0], share_planes, nsample[0]) # fusion p2 and p1

        self.seg = nn.Sequential(nn.Linear(planes[0], planes[0]),
                                 BatchNorm1d_P(planes[0]),
                                 Swish(),
                                 nn.Linear(planes[0], num_classes))

        self.embedf = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(embed_dim, embed_dim))

    def _make_enc(self, block, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(blocks):     # different from PointTransformerCls
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes, nsample, is_head=False):
        layers = [TransitionUp(self.in_planes, None if is_head else planes, self.global_pooling if is_head else None)]
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def _get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, t):
        """
        :param inputs: (b,3,n)
        :param t: (b,)
        :return: (b,3,n)
        """
        px = inputs.transpose(1, 2)  # (b,n,3)
        p0, x0 = px[..., :3].contiguous(), px.contiguous()   # coords, features
        temb = self.embedf(self._get_timestep_embedding(t, inputs.device))[:, :, None].expand(-1, -1, inputs.shape[-1])  # time embedding
        t0 = temb.transpose(1, 2) # (b,n,64)

        p1, x1, t1 = self.enc1([p0, x0, t0])
        p2, x2, t2 = self.enc2([p1, x1, t1])
        p3, x3, t3 = self.enc3([p2, x2, t2])
        p4, x4, t4 = self.enc4([p3, x3, t3])
        p5, x5, t5 = self.enc5([p4, x4, t4])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5]), t5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4], [p5, x5]), t4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3], [p4, x4]), t3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2], [p3, x3]), t2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1], [p2, x2]), t1])[1]

        res = self.seg(x1)
        return res


class PointTransformerSeg26(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg26, self).__init__(Bottleneck, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], **kwargs)


class PointTransformerSeg39(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg39, self).__init__(Bottleneck, [1, 2, 2, 2, 2], [1, 1, 2, 2, 1], **kwargs)


class PointTransformerSeg42(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg42, self).__init__(Bottleneck, [1, 1, 2, 2, 1], [1, 2, 2, 2, 2], **kwargs)


class PointTransformerSeg44(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg44, self).__init__(Bottleneck, [1, 2, 2, 2, 2], [1, 2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    stride = [1, 2, 4, 4, 4]
    nsample = [16, 32, 32, 32, 16]
    global_pooling = 'max'
    model = PointTransformerSeg42(in_channels=3, num_classes=3, embed_dim=64,
                                  stride=stride, nsample=nsample, global_pooling=global_pooling).cuda()
    x = torch.randn(16, 3, 2048).cuda()
    t = torch.randn(16,).cuda()
    res = model(x, t).transpose(1, 2)
    pass
