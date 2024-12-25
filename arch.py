import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        modules = []
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        return sum(res)


class Decoder(nn.Module):
    def __init__(self, in_channels, low_level_channels, out_channels):
        super(Decoder, self).__init__()

        self.conv_low_level = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn_low_level = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels + 48, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_low_level(low_level_feat)
        low_level_feat = self.bn_low_level(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.last_conv(x)

        return x


from torchvision.models import resnet50
class ImageTextDecoderClassifier(nn.Module):
    def __init__(self, image_dim, text_dim1, d_model, nhead, num_decoder_layers, dim_feedforward, dropout,num_output):
        super(ImageTextDecoderClassifier, self).__init__()

        self.image_transform = nn.Linear(image_dim, d_model)

        self.fuse_text1 = nn.Linear(text_dim1, d_model)



        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.classifier = nn.Linear(d_model, num_output)  

    def forward(self, image_features, text_features1):

        memory = self.image_transform(image_features).unsqueeze(0)

        fused_text1 = self.fuse_text1(text_features1)


        decoder_input = fused_text1
        decoder_input = decoder_input.unsqueeze(0)


        decoder_output = self.transformer_decoder(decoder_input, memory)



        logits = self.classifier(decoder_output.squeeze(0))

        return logits


class multitask(nn.Module):
    def __init__(self):
        super(multitask, self).__init__()


        self.backbone = models.resnet34(pretrained=True)

        del self.backbone.fc

        self.aspp = ASPP(512, 256, [6, 12, 18]) 

        self.decoder = Decoder(256, 256, 256)

        self.classifier = nn.Conv2d(256, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.subtype_classifier = nn.Linear(256,3)
        self.bm_classfication = nn.Linear(256,1)
        self.middle_featuremap_generater = nn.Conv2d(256,1,1)
        self.up_enhance = nn.Linear(1,256)

        self.embed_reshape = nn.Linear(512, 256)  
        self.text_decoder = ImageTextDecoderClassifier(image_dim=256, text_dim1=256, d_model=768,
                                                       dim_feedforward=2048, nhead=8, num_decoder_layers=3, dropout=0.4,
                                                       num_output=256)

        self.cam  = channel_attention(in_planes=256)
        self.sig_act = nn.Sigmoid()


    def forward(self, x,clip_encoded_img,coors):


        orig_size = x.size()[2:]


        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)

        middle_feature = self.middle_featuremap_generater(layer3)
        middle_feature = F.interpolate(middle_feature, size=orig_size, mode='bilinear', align_corners=True)

        x = self.backbone.layer4(layer3)
        x = self.aspp(x)
        avg = self.avgpool(x)
        flatten_feature = torch.flatten(avg,1)#3,256
        embed_coors = self.embed_reshape(coors)
        fuse_coors_feature = flatten_feature + embed_coors
        coordinates_output = self.text_decoder(fuse_coors_feature,clip_encoded_img)#3,256
        bn_output = coordinates_output
        coordinates_feature = coordinates_output.unsqueeze(-1).unsqueeze(-1)

        subtype = self.subtype_classifier(flatten_feature)


        x = x * coordinates_feature
        x = self.decoder(x, layer3)
        x = self.classifier(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)
        x = 0.8*x + 0.2 * middle_feature

        bn_classifer = self.bm_classfication(bn_output)
        return x,subtype,bn_classifer



if __name__ == "__main__":

    x = torch.randn(3,1)
    image = torch.randn(3,3,1400,1024)
    text = torch.randn(3,256)
    coors = torch.randn(3,512)




    model_seg = multitask()

    mask,subtype,bm = model_seg(image,text,coors)
    print(bm.shape)
