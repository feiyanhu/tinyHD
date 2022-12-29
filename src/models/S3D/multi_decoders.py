import torch
import os
import torch.nn as nn
from .S3D_featureExtractor import S3D_featureExtractor_multi_output, BasicConv3d
from .Decoders import Decoder2, Decoder3, Decoder4, Decoder5

__all__ = ['SalGradNet']

class BasicConv3d_rm(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d_rm, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        #self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = self.relu(x)
        return x

class SepConv3d_rm(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d_rm, self).__init__()
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
        self.last_conv = nn.Conv2d(6, 1, kernel_size=1, stride=1)
        #for tmp in [self.conv_t5, self.decoder5, self.conv_t4, self.decoder4, self.conv_t3_1, self.conv_t3_2, self.decoder3, self.conv_t2_1, self.conv_t2_2, self.conv_t2_3, self.decoder2, self.last_conv]:
        #    del tmp
        
        self.adp1 = BasicConv3d_rm(192, 480, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.adp2 = BasicConv3d_rm(480, 832, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.adp3 = BasicConv3d_rm(832, 1024, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.adp3_1 = BasicConv3d_rm(120, 120, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.adp3_2 = BasicConv3d_rm(208, 208, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.adp3_3 = BasicConv3d_rm(256, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        self.adp4 = BasicConv3d_rm(120, 208, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.adp5 = BasicConv3d_rm(208, 256, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

        self.adp5_1 = BasicConv3d_rm(52, 52, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.adp5_2 = BasicConv3d_rm(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.adp6 = BasicConv3d_rm(52, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.adp7 = BasicConv3d_rm(16, 1, kernel_size=(2,3,3), stride=(1,1,1), padding=(0,1,1))
    
    def add_unet(self):
        self.u_adp1_1 = BasicConv3d_rm(1024, 832, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.u_adp1_2 = BasicConv3d_rm(208, 208, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.u_adp2_1 = BasicConv3d_rm(208, 832, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.u_adp2_2 = BasicConv3d_rm(120, 480, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.u_adp2_3 = BasicConv3d_rm(48, 192, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        self.u_adp3_1 = BasicConv3d_rm(208, 480, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.u_adp3_2 = BasicConv3d_rm(120, 120, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.u_adp3_3 = BasicConv3d_rm(120, 120, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.u_adp3_4 = BasicConv3d_rm(48, 48, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

        self.u_adp4_1 = BasicConv3d_rm(30, 48, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.u_adp4_2 = BasicConv3d_rm(12, 12, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

        self.u_adp5_1 = BasicConv3d_rm(12, 1, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
    def forward_d3(self, features):
        x1 = self.adp1(features[0])
        x2 = self.adp2(features[1])
        x3 = self.adp3(features[2])
        x1 = x1 + self.trilinear_up2(features[1])
        x2 = x2 + self.trilinear_up2(features[2])
        x3 = x3 + features[3]
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
        return out.squeeze(1)

    def forward_d2(self, features):
        #features[:-1] = [self.upsample_features(tmp) for tmp in features[:-1]]
        #for y in features:print(y.shape)
        y = self.upsample_features(self.u_adp1_1(features[-1])) + self.u_adp1_2(self.upsample_features(features[-2]))
        
        y = self.upsample_features(self.u_adp2_1(y))
        

        y = self.upsample_features(self.u_adp3_1(y)) + self.u_adp3_2(self.upsample_features(self.u_adp2_2(self.upsample_features(features[-3]))))
        
        y = self.upsample_features(self.u_adp3_3(y))
        
        y = self.upsample_features(self.u_adp4_1(y)) + self.u_adp4_2(self.upsample_features(self.u_adp3_4(self.upsample_features(self.u_adp2_3(self.upsample_features(features[-4]))))))
        
        y = self.u_adp5_1(y)
        #print(y.shape)
        #exit()
        return y.squeeze(1)

    def forward_d1(self, features):
        #DECONV 2
        x2 = self.conv_t2_1(features[0])
        x2 = self.conv_t2_2(x2)
        x2 = self.conv_t2_3(x2)
        sal2 = self.decoder2(x2.squeeze(2))

        #DECONV 3
        x3 = self.conv_t3_1(features[1])
        x3 = self.conv_t3_2(x3)
        sal3 = self.decoder3(x3.squeeze(2))

        #DECONV 4
        x4 = self.conv_t4(features[2])
        sal4 = self.decoder4(x4.squeeze(2))

        #DECONV 5
        x5 = self.conv_t5(features[3])
        sal5=self.decoder5(x5.squeeze(2))
        
        x = torch.cat((sal2, sal3, sal4, sal5), 1)
        #x = self.last_conv(x)
        #print(x.shape)
        #exit()
        return x

    def forward(self, x):
        _, features2, features3, features4, features5 = self.featureExtractor(x)
        #print(features5.shape, features4.shape, features3.shape, features2.shape)
        features = [features2, features3, features4, features5]

        map1 = self.forward_d1(features) #no sigmoid
        map2 = self.sigmoid(self.forward_d2(features))
        map3 = self.sigmoid(self.forward_d3(features))
        #print(map1.shape, map2.shape, map3.shape)
        out = torch.cat([map1, map2, map3], dim=1)
        out = self.sigmoid(self.last_conv(out))
        #return out, [map1, 0, 0]
        return out, [map1, map2, map3]

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