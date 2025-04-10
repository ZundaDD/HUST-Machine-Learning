import warnings
from torch.utils.data import DataLoader
from data import *
import torch
from tqdm import tqdm
import os
import argparse
import pandas as pd
warnings.filterwarnings('ignore')

def access(model_path, epoch):

    model, reshape = get_model(args.model)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model = model.to(device=device)

    model.eval()
    with torch.no_grad():
        running_corrects = 0
        for (images, labels, paths) in tqdm(dataloader):
            images = images.to(device)
            images = reshape(images)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(preds)):
                pred = preds[i]
                actual = int(labels[i])

                if actual == pred:
                    running_corrects += 1

        result[int(epoch / 5) - 1] += (running_corrects * 100) / len(dataset)

    del model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='resnet18', type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")

    # load dataset
    image_dir = os.path.join("dataset", "test")
    dataset = EasyDataset(image_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4, drop_last=False)

    epochs = [i * 5 for i in range(1,21)]
    result = [0 for i in range(len(epochs))]

    model_dir = f"result/{args.seed}"
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".pth"):
            epoch = int(file_path.split('_')[-1].split('.')[0])
            if epoch in epochs:
                access(file_path, epoch)


    df = pd.DataFrame(result)
    df.to_csv(f'result/{args.seed}/result.csv')