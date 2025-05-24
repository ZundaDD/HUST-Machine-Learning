import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision import transforms as T
from data import EasyDataset
from model import get_model
import warnings
warnings.filterwarnings(action='ignore')

class Learner(object):
    def __init__(self, args):
        self.args = args

        self.trans = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Normalize((0.1307,), (0.3081,))
            ])

        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
            
        self.train_dataset = EasyDataset(os.path.join(args.data_dir,"train"))
        self.test_dataset = EasyDataset(os.path.join(args.data_dir,"test"))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.n_classes = 10
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.log_dir = os.path.join(args.log_dir, f"{args.index}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_acc = 0
        self.acc = {"train":[], "test":[]}

        self.model, self.reshape = get_model(args.model, args.pretrained)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        print('Finished model initialization....')


    def evaluate(self, model, data_loader):
        model.eval()

        pred_ls = torch.tensor([]).to(self.device)
        label_ls = torch.tensor([]).to(self.device)

        for data, label, index in data_loader:
            label = torch.tensor(label).to(self.device)
            data = self.reshape(data).to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)

                pred_ls = torch.cat((pred_ls, pred), 0)
                label_ls = torch.cat((label_ls, label), 0)

        model.train()
        correct = (pred_ls == label_ls).sum().item()
        return correct / len(data_loader.dataset) * 100
    
    def save_best(self, epoch):
        model_path = os.path.join(self.log_dir, f"{self.args.seed}_model_best.pt")
        state_dict = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{epoch} step model saved ...')

    def board_acc(self, epoch):
        train_acc = self.evaluate(self.model, self.train_loader)
        test_acc = self.evaluate(self.model, self.test_loader)

        print(f'epoch: {epoch}\ttrain: {train_acc:.5f}%\ttest: {test_acc:.5f}%')
        self.acc["train"].append(train_acc)
        self.acc["test"].append(test_acc)

        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            self.save_best(epoch)

    def train(self, args):

        for epoch in range(args.epochs):
            for data, label, index in self.train_loader:

                data = self.reshape(data).to(self.device)
                if args.augment:
                    data = self.trans(data)
                label = torch.tensor(label).to(self.device)

                logit = self.model(data)
                loss_update = self.criterion(logit, label)
                loss = loss_update.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.board_acc(epoch)

        print('Finished training')
        print(f'best_test_acc:{self.best_acc:.5f}%')

        df = pd.DataFrame(self.acc)
        df.to_csv(os.path.join(self.log_dir, f"{self.args.seed}_result.csv"))