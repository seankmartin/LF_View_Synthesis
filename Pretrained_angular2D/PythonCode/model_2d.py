import torch.nn as nn
import common

class C2D(nn.Module):
    def __init__(
        self, args, inchannels, outchannels, conv=common.default_conv):
        super(C2D, self).__init__()
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        #Originally used ReLU
        act = nn.ELU(inplace=True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std, channels=3 * inchannels)
        

        # define head module
        m_head = [conv(inchannels * 3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            nn.Conv2d(
                n_feats, outchannels * 3, kernel_size,
                padding=(kernel_size//2)
            ),
            nn.Tanh()
        ]

        self.add_mean = common.MeanShift(
            1, rgb_mean, rgb_std, channels=3 * outchannels, sign=1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, freeze=False, first=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                #Only copy the body information
                if not first:
                    if name.find('body') is not -1:
                        param = param.data
                        print("Loading", name)
                        try:
                            own_state[name].copy_(param)
                        except Exception:
                            if name.find('tail') == -1:
                                raise RuntimeError('While copying the parameter named {}, '
                                                'whose dimensions in the model are {} and '
                                                'whose dimensions in the checkpoint are {}.'
                                                .format(name, own_state[name].size(), param.size()))
                else:
                    param = param.data
                    print("Loading", name)
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        if name.find('tail') == -1:
                            raise RuntimeError('While copying the parameter named {}, '
                                            'whose dimensions in the model are {} and '
                                            'whose dimensions in the checkpoint are {}.'
                                            .format(name, own_state[name].size(), param.size()))
        if freeze:
            for name, param in self.named_parameters():
                if name.find('body') is not -1:
                    param.requires_grad = False
        