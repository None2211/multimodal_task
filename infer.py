import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import torchvision.utils as vutils
import numpy as np
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch import Tensor
from PIL import Image

from albumentations import Compose, RandomCrop
from albumentations.pytorch import ToTensorV2
from dataset import MultitaskDataset

from deelab_seg import multitask
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.utils as vutils
import pandas as pd
from sklearn.metrics import roc_auc_score
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)
best_model_weight= '...'

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()
def test_model(model, data_loader, device):
    mask_values =  [0, 1]
    model.load_state_dict(torch.load(best_model_weight))
    model.eval()
    results_seg = []
    all_bm_probs = []
    all_side_labels = []
    with torch.no_grad():
        correct_subtype_predictions = 0
        correct_bm_predictions = 0

        total_samples = 0
        progress_bar = tqdm(data_loader)
        for i, batch in enumerate(progress_bar):
            inputs = batch['image'].to(dtype=torch.float32).to(device)
            name = batch['idx']
            masks = batch['mask'].to(dtype=torch.float32).to(device)
            subtype_label = batch['sub_label'].squeeze(1).to(dtype=torch.long, device=device)
            type_label = batch['type_label'].to(dtype=torch.float32, device=device)
            text = batch['img_clip_feature'].to(dtype=torch.float32, device=device)

            coors = batch['centroid_center'].to(dtype=torch.float32, device=device)

            output, subtype_out, bm_out = model(inputs, text,coors)

            preds = (torch.sigmoid(output) > 0.5)
            assert torch.all((masks == 0) | (masks == 1)), 'Mask contains values other than 0 and 1'
            dice = dice_coeff(preds, masks).item()

            _, predicted_categories = subtype_out.max(1)
            correct_subtype_predictions += (predicted_categories == subtype_label).sum().item()

            bm_preds = (torch.sigmoid(bm_out) > 0.5)
            correct_bm_predictions += (bm_preds.squeeze() == type_label).sum().item()
            bm_probs = torch.sigmoid(bm_out).squeeze()

            all_bm_probs.append(bm_probs.item())  
            all_side_labels.append(side_label.item())

            total_samples += side_label.size(0)
            progress_bar.set_postfix({
                'Dice': dice,
                'Batch Category Acc': (predicted_categories == subtype_label).float().mean().item(),
                'Batch BM Acc': (bm_preds.squeeze() == side_label).float().mean().item()
            })



            predsmap = preds.cpu()
            predsmap = predsmap[0].long().squeeze().numpy()
            print(predsmap.shape)
            result = mask_to_image(predsmap, mask_values)
            result_name = os.path.basename(name[0])
            desired_name = result_name.split('_', 1)[0] + result_name[result_name.rfind('_'):]
            save_folder = r'...'
            full_save_path = os.path.join(save_folder, desired_name)
            result.save(full_save_path)  

            results_seg.append((name, dice))


        avg_category_acc = correct_subtype_predictions / total_samples
        avg_bm_acc = correct_bm_predictions / total_samples

        dice_all = np.mean([score for _, score in results_seg])
        df = pd.DataFrame(results_seg, columns=['Name', 'Dice Score'])
        print(f"Avg Dice: {dice_all:.4f}")
        print(f"Avg Category Accuracy: {avg_category_acc:.4f}")
        print(f"Avg BM Accuracy: {avg_bm_acc:.4f}")


        df.to_csv('dice_scores_text.txt', sep='\t', index=False)
        auc_score = roc_auc_score(all_side_labels, all_bm_probs)
        print(f"AUC: {auc_score:.4f}")

        return results_seg

if __name__ == "__main__":
    with open(r'test_filenames_small_f.txt', 'r') as file:
        test_filenames = file.read().splitlines()

    image_folder = r'...'
    mask_folder = r'...'

    img_folder_224 = r'...'
    centroids_filepath = r"..."
    medi_pretrain = r'...'

    test_dataset = MultitaskDataset(image_folder,medi_pretrain,img_folder_224,mask_folder, test_filenames,centroids_filepath=centroids_filepath)
    train_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = multitask().to(device)




    dice_scores = test_model(model,train_loader, device)
