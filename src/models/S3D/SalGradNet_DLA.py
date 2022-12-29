import torch
import os
import torch.nn as nn
from .S3D_featureExtractor import S3D_featureExtractor_multi_output#, BasicConv3d
from .Decoders import Decoder2, Decoder3, Decoder4, Decoder5

__all__ = ['SalGradNet']

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        #self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = self.relu(x)
        return x

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class SalGradNet(nn.Module):
    def __init__(self, pretrained=False):
        super(SalGradNet, self).__init__()
        self.featureExtractor=S3D_featureExtractor_multi_output()        
        
        if pretrained:
            print('Loading weights...')
            #weight_dict=torch.load(os.path.join('models','S3D_kinetics400.pt'))
            s3d_path = '../pretrained/S3D_kinetics400.pt'
            weight_dict=torch.load(s3d_path)

            model_dict=self.featureExtractor.state_dict()
            
            list_weight_dict=list(weight_dict.items())
            list_model_dict=list(model_dict.items())
            
            for i in range(len(list_model_dict)):
                assert list_model_dict[i][1].shape==list_weight_dict[i][1].shape
                model_dict[list_model_dict[i][0]].copy_(weight_dict[list_weight_dict[i][0]])
            
            for i in range(len(list_model_dict)):
                assert torch.all(torch.eq(model_dict[list_model_dict[i][0]],weight_dict[list_weight_dict[i][0]].to('cpu')))
            print('Loading done!')
        
        self.shuffle_up2 = nn.PixelShuffle(2)
        self.trilinear_down = nn.Upsample(scale_factor=(0.5, 1, 1), mode='trilinear', align_corners=True)
        self.trilinear_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def upsample_features(self, features, temporal_upsample=False):
        N, C, T, H, W = features.shape
        features = features.permute(0, 2, 1, 3, 4).reshape(N*T, C, H, W)
        features = self.shuffle_up2(features)
        features = features.view(N, T, int(C/4), 2*H, 2*W).permute(0, 2, 1, 3, 4)
        if temporal_upsample:
            features = self.bilinear_up2(features)
        return features
    
    def add_original(self):
        #conv_t: T/8-->1
        self.conv_t5 = BasicConv3d(1024, 1024, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder5=Decoder5(1024, out_sigmoid=True)
        
         #conv_t: T/8-->1
        self.conv_t4 = BasicConv3d(832, 832, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder4=Decoder4(832, out_sigmoid=True)
        
         #conv_t: T/4-->1
        self.conv_t3_1 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t3_2 = BasicConv3d(480, 480, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder3=Decoder3(480, out_sigmoid=True)
        
        #conv_t: T/2-->1
        self.conv_t2_1 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_2 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_3 = BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.decoder2=Decoder2(192, out_sigmoid=True)
        
        self.last_conv = nn.Conv2d(4, 1, kernel_size=1, stride=1)
    def add_layer(self):
        #for tmp in [self.conv_t5, self.decoder5, self.conv_t4, self.decoder4, self.conv_t3_1, self.conv_t3_2, self.decoder3, self.conv_t2_1, self.conv_t2_2, self.conv_t2_3, self.decoder2, self.last_conv]:
        #    del tmp
        
        self.adp1 = BasicConv3d(192, 480, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.adp2 = BasicConv3d(480, 832, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.adp3 = BasicConv3d(832, 1024, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.adp3_1 = BasicConv3d(120, 120, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.adp3_2 = BasicConv3d(208, 208, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.adp3_3 = BasicConv3d(256, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        self.adp4 = BasicConv3d(120, 208, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.adp5 = BasicConv3d(208, 256, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

        self.adp5_1 = BasicConv3d(52, 52, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.adp5_2 = BasicConv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.adp6 = BasicConv3d(52, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.adp7 = BasicConv3d(16, 1, kernel_size=(2,3,3), stride=(1,1,1), padding=(0,1,1))
    
    def forward(self, x):
        _, features2, features3, features4, features5 = self.featureExtractor(x)
        #print(features5.shape, features4.shape, features3.shape, features2.shape)
        x1 = self.adp1(features2)
        x2 = self.adp2(features3)
        x3 = self.adp3(features4)
        x1 = x1 + self.trilinear_up2(features3)
        x2 = x2 + self.trilinear_up2(features4)
        x3 = x3 + features5
        #[x1, x2, x3] = [self.trilinear_up2(tmp) for tmp in [x1, x2, x3]]
        [x1, x2, x3] = [self.upsample_features(tmp) for tmp in [x1, x2, x3]]
        [x1, x2, x3] = [m(tmp) for tmp, m in zip([x1, x2, x3], [self.adp3_1, self.adp3_2, self.adp3_3])]

        y1 = self.adp4(x1)
        y2 = self.adp5(x2)
        y1 = y1 + self.trilinear_up2(x2)
        y2 = y2 + self.trilinear_up2(x3)
        #[y1, y2] = [self.trilinear_up2(tmp) for tmp in [y1, y2]]
        [y1, y2] = [self.upsample_features(tmp) for tmp in [y1, y2]]
        [y1, y2] = [m(tmp) for tmp, m in zip([y1, y2], [self.adp5_1, self.adp5_2])]
        #print(y1.shape, y2.shape)
        #exit()

        z1 = self.adp6(y1)
        z1 = z1 + self.trilinear_up2(y2)
        #z1 = self.trilinear_up2(z1)
        z1 = self.upsample_features(z1)
        
        out = self.adp7(z1)
        out = self.sigmoid(out)
        #print(out.shape)
        #exit()
        
        #print(sal5.shape, sal4.shape, sal3.shape, sal2.shape)
        #print(x5.squeeze(2).shape, x4.squeeze(2).shape)
        #print(features5.shape, features4.shape, features3.shape, features2.shape)
        #exit()
        #return  features5, features4, features3, features2, out.squeeze(1)
        return  out.squeeze(1).squeeze(1)
        #return  0, 0, 0, 0, out.squeeze(1)

    def get_optimizer(self, lr, use_adam=False, exclude_list=[None]):
        weight_decay_default = 0.0005
        lr_dict = {'featureExtractor': 0.5*lr}
        regularization_dict = {'featureExtractor': 0.0005}
        params = []
        parem_dict = dict(self.named_parameters())
        #parem_dict.update(dict(self.student_audio.named_parameters()))
        #print(len(parem_dict))
        for key, value in parem_dict.items():
            module_name = key.split('.')[0]
            if value.requires_grad and module_name not in exclude_list:
                if module_name in lr_dict.keys():
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr_dict[module_name] * 2, 'weight_decay': 0}]
                        print(key, lr_dict[module_name] * 2)
                    else:
                        params += [{'params': [value], 'lr': lr_dict[module_name],
                                    'weight_decay': regularization_dict[module_name]}]
                        print(key, lr_dict[module_name])
                else:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                        print(key, lr * 2)
                    else:
                        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay_default}]
                        print(key, lr)
        #exit()
        if use_adam:
            optimizer = torch.optim.Adam(params)
        else:
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        return optimizer