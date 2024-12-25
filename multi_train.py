
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
import numpy as np
import random
from CFG import CFG



from dataset import MultitaskDataset

from arch import multitask
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import f1_score
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.85, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
def dice_coefficient(preds, targets):
    smooth = 1e-6
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps  

    def forward(self, output, target):

        output = torch.sigmoid(output)



        intersection = (output * target).sum()


        dice_coeff = (2. * intersection) / (output.sum() + target.sum() + self.eps)

        # Dice loss
        dice_loss = 1 - dice_coeff
        print("Output Min:", output.min().item(), "Max:", output.max().item())
        print("Target Min:", target.min().item(), "Max:", target.max().item())
        print("Intersection:", intersection.item())
        print("Dice Coefficient:", dice_coeff.item())
        print("Output Sum:", output.sum().item())
        print("Target Sum:", target.sum().item())

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):

        assert output.size() == target.size(),'channel is not equal'
        output = torch.sigmoid(output)


        intersection = (output * target).sum()


        union = output.sum() + target.sum() - intersection + self.eps

        iou = intersection / union


        return 1 - iou

def train_oneepoch_model(model_seg, data_loader, optimizer_seg, device):

    model_seg.train()


    total_seg_loss = 0
    total_combined_loss = 0
    total_category_loss = 0
    total_bm_loss = 0






    progress_bar = tqdm(data_loader)
    for batch in progress_bar:
        inputs = batch['image'].to(dtype=torch.float32).to(device)
        assert inputs.min() >= 0 and inputs.max() <= 1, 'True mask indices should be in [0, 1]'


        masks = batch['mask'].to(dtype=torch.float32).to(device)
        text = batch['img_clip_feature'].to(dtype=torch.float32,device=device)


        type_label = batch['type_label'].to(dtype=torch.float32,device=device)
        subtype_label = batch['sub_label'].squeeze(1).to(dtype=torch.long, device=device)

        coos_label = batch['centroid_center'].to(dtype=torch.float32, device=device)



        criterion_seg = IoULoss()
        criterion_category = nn.CrossEntropyLoss()
        criterion_bm = nn.BCEWithLogitsLoss()

        output, subtype_out, bm_out = model_seg(inputs,text,coos_label)

        assert torch.all((masks == 0) | (masks == 1)), 'Mask contains values other than 0 and 1'

        loss_seg_deep = criterion_seg(output, masks)


        loss_seg = loss_seg_deep
        loss_sub = criterion_category(subtype_out, subtype_label)
        loss_bm = criterion_bm(bm_out, type_label)

        total_segsubloss = (loss_sub + loss_seg + loss_bm) / 3

        total_seg_loss += loss_seg.item()
        total_category_loss += loss_sub.item()
        total_bm_loss += loss_bm.item()
        total_combined_loss = (total_seg_loss + total_category_loss + total_bm_loss) / 3

        optimizer_seg.zero_grad()
        total_segsubloss.backward()
        optimizer_seg.step()



        progress_bar.set_description(f"seg_loss:{loss_seg.item()}, sub_loss:{loss_sub.item()},bm_loss:{loss_bm.item()}")

    avg_seg_loss = total_seg_loss / len(data_loader)
    avg_category_loss = total_category_loss / len(data_loader)
    avg_bm_loss = total_bm_loss / (len(data_loader))
    avg_total_loss = total_combined_loss / len(data_loader)






    wandb.log({"Segmentation Loss": avg_seg_loss,"sub Loss": avg_category_loss,"bm_loss":avg_bm_loss,"combined_loss":avg_total_loss})

    print(f"Segmentation Loss: {avg_seg_loss:.4f},combine Loss: {avg_total_loss:.4f},cateory loss:{avg_category_loss:.4f},bm loss:{avg_bm_loss}")


def validate_oneepoch_model(model_seg, data_loader, device):

    model_seg.eval()

    total_seg_loss = 0
    total_combined_loss = 0
    total_category_loss = 0
    total_bm_loss = 0




    with torch.no_grad():
        total_dice = 0.0

        correct_subtype_predictions = 0
        correct_bm_predictions = 0

        total_samples = 0
        progress_bar = tqdm(data_loader)
        for batch in progress_bar:
            inputs = batch['image'].to(dtype=torch.float32).to(device)

            masks = batch['mask'].to(dtype=torch.float32).to(device)
            text = batch['img_clip_feature'].to(dtype=torch.float32, device=device)
            subtype_label = batch['sub_label'].squeeze(1).to(dtype=torch.long, device=device)
            type_label = batch['type_label'].to(dtype=torch.float32,device=device)

            coos_label = batch['centroid_center'].to(dtype=torch.float32, device=device)



            criterion_seg = IoULoss()
            criterion_category = nn.CrossEntropyLoss()
            criterion_bm = nn.BCEWithLogitsLoss()






            output,subtype_out,bm_out = model_seg(inputs,text,coos_label)

            assert torch.all((masks == 0) | (masks == 1)), 'Mask contains values other than 0 and 1'

            loss_seg_deep = criterion_seg(output, masks)


            loss_seg = loss_seg_deep
            loss_sub = criterion_category(subtype_out, subtype_label)
            loss_bm = criterion_bm(bm_out, type_label)
            total_segsubloss = (loss_sub + loss_seg + loss_bm) / 3


            total_seg_loss += loss_seg.item()
            total_category_loss += loss_sub.item()
            total_bm_loss += loss_bm.item()
            total_combined_loss = (total_seg_loss + total_category_loss + total_bm_loss) / 3
            assert masks.min() >= 0 and masks.max() <= 1, 'True mask indices should be in [0, 1]'
            preds = (torch.sigmoid(output) > 0.5)
            dice = dice_coefficient(preds, masks)
            total_dice += dice.item()

            _, predicted_categories = subtype_out.max(1)
            correct_subtype_predictions += (predicted_categories == subtype_label).sum().item()
            bm_preds = (torch.sigmoid(bm_out) > 0.5)
            correct_bm_predictions += (bm_preds == side_label ).sum().item()



            total_samples += side_label.size(0)

            progress_bar.set_postfix({
                'Dice': dice.item(),
                'Batch Category Acc': (predicted_categories == subtype_label).float().mean().item(),
                'Batch bm acc':(bm_preds == side_label ).float().mean().item()
            })

    avg_dice = total_dice / len(data_loader)
    avg_category_acc = correct_subtype_predictions / total_samples
    avg_bm_acc = correct_bm_predictions / total_samples

    avg_seg_loss = total_seg_loss / len(data_loader)
    avg_category_loss = total_category_loss / len(data_loader)
    avg_bm_loss = total_bm_loss / len(data_loader)
    avg_combined_loss = total_combined_loss / len(data_loader)



    wandb.log({
        "Validation Dice Score": avg_dice,
        "Validation Category Acc": avg_category_acc,
        "VAL Segmentation Loss": avg_seg_loss,
        "VAL bm acc":avg_bm_acc,
        "Val bm loss":avg_bm_loss,
        "VAL Category Loss": avg_category_loss,
        "val combined loss": avg_combined_loss
    })

    print(f"Validation Dice Score: {avg_dice:.4f}")

    print(f"Validation Category Acc: {avg_category_acc:.4f}")
    print(f"Validation bm Acc: {avg_bm_acc:.4f}")

    return avg_dice,avg_category_acc,avg_bm_acc








def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description='Segmentation Training')
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, help='Mini-batch size')
    parser.add_argument('--lr_seg', default=0.002, type=float, help='Learning  rate seg')


    parser.add_argument('--load_seg', default=r'...', type=str, help='Path to load the pre-trained model')

    args = parser.parse_args()

    with open('train_filenames_small_f.txt', 'r') as file:
        train_filenames = file.read().splitlines()
    train_filenames, val_filenames = train_test_split(train_filenames, test_size=0.15, random_state=42)
    image_folder = r'...' # preprocessed image folder
    mask_folder = r'...' # mask folder
    img_folder_224 = r'...' # resized original image
    centroids_filepath = r"..." # coordinates path
    medi_pretrain = r'...' # pretrained alignment checkpoint path

    crop_height = 128
    crop_width = 128




    train_dataset = MultitaskDataset(image_folder,medi_pretrain,img_folder_224,mask_folder,train_filenames,centroids_filepath)
    val_dataset = MultitaskDataset(image_folder,medi_pretrain,img_folder_224,mask_folder,val_filenames,centroids_filepath)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    wandb.init(project="segmentation-learning")
    config = wandb.config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr_seg = args.lr_seg








    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_seg = multitask().to(device)




    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)




    optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.lr_seg, momentum=0.9)




    scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, 'max', patience=10, factor=0.1)


    best_combined = 0.0






    if args.load_seg:
        model_seg.load_state_dict(torch.load(args.load_seg))


    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_oneepoch_model(model_seg, train_loader, optimizer_seg,  device=device)
        avg_dice, avg_category_acc,avg_bm_acc = validate_oneepoch_model(model_seg, val_loader, device=device)

        combined_res = (avg_dice + avg_category_acc + avg_bm_acc) / 3


        scheduler_seg.step(combined_res)



        if combined_res > best_combined:
            best_combined = combined_res
            torch.save(model_seg.state_dict(), "medi_pre_124_34_fine.pth")
            print(f"New best segmentation model saved with Dice score: {best_combined:.4f}")


    print("Best Scores:")

    wandb.finish()

if __name__ == "__main__":
    main()
