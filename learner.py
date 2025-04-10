import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchvision import transforms as T
from data import EasyDataset, get_model
import warnings
warnings.filterwarnings(action='ignore')

class Learner(object):
    def __init__(self, args):
        self.args = args

        self.trans = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            ])

        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
            
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

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.best_acc = 0
        self.n_classes = 10

        self.model, self.reshape = get_model(args.model)

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
            data = data.to(self.device)
            data = self.reshape(data)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)

                pred_ls = torch.cat((pred_ls, pred), 0)
                label_ls = torch.cat((label_ls, label), 0)

        model.train()
        correct = 0
        for i in range(pred_ls.shape[0]):
            if pred_ls[i] == label_ls[i]:
                correct += 1
        return correct / len(data_loader.dataset) * 100
    
    def save_best(self, step):
        os.makedirs(os.path.join("result", f"{self.args.seed}"), exist_ok=True)
        model_path = os.path.join("result", f"{self.args.seed}","model_best.pt")
        state_dict = {
            'steps': step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{step} step model saved ...')

    def log_model(self, epoch):
        model_dir = os.path.join("result", f"{self.args.seed}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(model_dir, f'model_{epoch}.pth'))
        print(f'{epoch} epoch model saved ...')
        print('')

    def calculate_acc(self, step, epoch):
        train_acc = self.evaluate(self.model, self.train_loader)
        test_acc = self.evaluate(self.model, self.test_loader)

        print(f'epoch: {epoch}')
        print(f'train: {train_acc}% test: {test_acc}%')
        print('')

        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            self.save_best(step)


    def train(self, args):
        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset)
        epoch, cnt = 0, 0

        for step in range(args.num_steps):
            try:
                data, label, index = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                data, label, index = next(train_iter)

            data = data.to(self.device)
            data = self.reshape(data)
            if args.augment:
                data = self.trans(data)
            label = torch.tensor(label)
            label = label.to(self.device)

            logit = self.model(data)
            loss_update = self.criterion(logit, label)
            loss = loss_update.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % args.calculate_freq == 0:
                self.calculate_acc(step, epoch)

            cnt += len(index)
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                print('')
                epoch += 1
                if epoch % args.log_freq == 0:
                    self.log_model(epoch)
                cnt = 0
