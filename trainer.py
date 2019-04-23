import torch as T
import torch.nn as nn
import numpy as np
import time
from sklearn import metrics
from os import path


class Trainer:
    def __init__(self, training_set,
                 validation_set,
                 batch_size,
                 n_epochs,
                 model,
                 optimizer,
                 loss,
                 name,
                 device='cuda',
                 deterministic=False,
                 parallel=False
                 ):

        T.backends.cudnn.deterministic = deterministic
        self.batch_size = batch_size
        self.dataset = training_set
        self.valid_dataset = validation_set

        self.model = model
        self.device = device
        self.parallel = parallel
        if parallel:
            self.model = nn.DataParallel(model)
        self.model.cuda(self.device)
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.name = name
        self.log = ''


    def train_epoch(self, epoch):
        s_time = time.time()
        self.model.train()
        all_losses = []
        all_acc = []
        for data, target in self.dataset:
            data, target = data.cuda(self.device), target.cuda(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            acc = self.calc_accuracy(output, target)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            all_losses.append(loss.item())
            all_acc.append(acc.cpu())

        valid_acc = self.validate()
        self.report(all_losses, all_acc, valid_acc, epoch, time.time() - s_time)

    def report(self, all_losses, all_acc, valid_acc, epoch, duration):
        n_train = len(all_losses)
        loss = np.sum(all_losses) / n_train

        def summery(data):
            n = 0.0
            s_dist = 0
            for dist in data:
                s_dist += T.sum(dist)
                n += len(dist)

            return s_dist.float() / n

        tr_dist = summery(all_acc)
        va_dist = summery(valid_acc)

        pred, target = self.predict()
        fpr, tpr, thresholds = metrics.roc_curve(target, pred)
        auc = metrics.auc(fpr, tpr)

        msg = f'epoch {epoch}: loss {loss:.3f} Tr Acc {tr_dist:.2f} Val Acc {va_dist:.2f} AUC {auc:.2f} duration {duration:.2f}'
        print(msg)
        self.log += msg + '\n'


    def predict(self):
        self.model.eval()
        all_pred = T.zeros(len(self.valid_dataset.dataset))
        all_targets = T.zeros(len(self.valid_dataset.dataset))
        for batch_idx, (data, target) in enumerate(self.valid_dataset):
            with T.no_grad():
                data, target = data.cuda(self.device), target.cuda(self.device)
                output = self.model(data)
            st = batch_idx * self.batch_size

            all_pred[st:st + output.shape[0]] = output.cpu().squeeze()
            all_targets[st:st + output.shape[0]] = target.cpu().squeeze()

        all_pred = all_pred.view(-1, 3).mean(dim=1)
        all_targets = all_targets.view(-1, 3).mean(dim=1)
        return all_pred, all_targets


    def validate(self):
        all_pred, all_targets = self.predict()
        matches = self.calc_accuracy(all_pred, all_targets)
        return [matches]

    def calc_accuracy(self, x, y):
        x_th = (x > 0.5).long()
        matches = x_th == y.long()
        return matches

    def run(self):
        start_t = time.time()
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
        diff = time.time() - start_t
        print(f'took {diff} seconds')
        with open(path.join('results',f'{self.name}.txt'),'w') as f:
            f.write(self.log)


