import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image
from datasets.cuhkpedes import CUHKPEDES


config_file  = '/xxx/configs.yaml'
args = load_train_configs(config_file)
args.batch_size = 1024
args.training = False
device = "cuda"
test_img_loader, test_txt_loader = build_dataloader(args)
model = build_model(args)
checkpointer = Checkpointer(model)
checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
model.to(device)

evaluator = Evaluator(test_img_loader, test_txt_loader)

qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())
qfeats = F.normalize(qfeats, p=2, dim=1) # text features
gfeats = F.normalize(gfeats, p=2, dim=1) # image features

similarity = qfeats @ gfeats.t()
# acclerate sort with topk
_, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)  # q * topk

dataset = CUHKPEDES(root='./data')
test_dataset = dataset.test

img_paths = test_dataset['img_paths']
captions = test_dataset['captions']
gt_img_paths = test_dataset['gt_img_paths']

def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    image_paths = [img_paths[j] for j in indices[idx]]
    image_ids = gids[indices[idx]]
    gt_image_path = gt_img_paths[idx]
    return query_id, image_ids, query_caption, image_paths, gt_image_path

def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path, fname=None):
    print(query_id)
    print(image_ids)
    print(query_caption)
    fig = plt.figure()
    col = len(image_paths)

    # plot ground truth image
    plt.subplot(1, col+1, 1)
    img = Image.open(gt_img_path)
    img = img.resize((128, 256))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    for i in range(col):
        plt.subplot(1, col+1, i+2)
        img = Image.open(image_paths[i])

        bwith = 2  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        if image_ids[i] == query_id:
            ax.spines['top'].set_color('lawngreen')
            ax.spines['right'].set_color('lawngreen')
            ax.spines['bottom'].set_color('lawngreen')
            ax.spines['left'].set_color('lawngreen')
        else:
            ax.spines['top'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['bottom'].set_color('red')
            ax.spines['left'].set_color('red')
        
        img = img.resize((128, 256))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    

    fig.show()
    if fname:
        plt.savefig(fname, dpi=300)
# idx is the index of qids(A list of query ids, range from 0 - len(qids))
query_id, image_ids, query_caption, image_paths, gt_img_path = get_one_query_caption_and_result_by_id(0, indices, qids, gids, captions, img_paths, gt_img_paths)
plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path)