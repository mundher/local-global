from trainer import Trainer
from preprocessing import get_dataset
from os import path
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import random
from sklearn import metrics
from os import path

def get_metrics(target, pred):
    prec, recall, _, _ = metrics.precision_recall_fscore_support(target, pred>0.5, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    return prec, recall, auc


def calc_accuracy(x, y):
    x_th = (x > 0.5).long()
    matches = x_th == y.long()
    return matches

def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def kfold(src_path,
          batch_size,
          n_epochs,
          model_optimizer,
          loss,
          name,
          device,
          deterministic=False,
          parallel=False,
          dataset_func=get_dataset):

    print(f'Experiment {name}')
    all_pred = T.zeros(849)
    all_targets = T.zeros(849)
    i =0
    f = open(path.join('results', f'{name}.txt'), 'w')
    f.write(f'{batch_size} {n_epochs} {model_optimizer}\n')
    for fold in range(10):

        reset_rand()

        print(f'------------ fold {fold+1} ------------')
        f.write(f'------------ fold {fold+1} ------------\n')
        trset, testset = dataset_func(path.join(src_path,str(fold)))
        print(f'Training Size: {len(trset)}, Validation Size: {len(testset)}')
        trset = DataLoader(trset, batch_size, shuffle=True)
        testset = DataLoader(testset, batch_size, shuffle=False)
        model,optimizer = model_optimizer()
        tr = Trainer(
            trset,
            testset,
            batch_size,
            n_epochs,
            model,
            optimizer,
            loss,
            f'{name}_{fold}',
            device,
            deterministic,
            parallel
        )

        tr.run()

        pred, target = tr.predict()
        all_pred[i:i+pred.shape[0]] = pred
        all_targets[i:i+target.shape[0]] = target
        i += target.shape[0]

        prec, recall, auc = get_metrics(target, pred)
        print(f'AUC: {auc}, precession: {prec}, Recall: {recall}')
        f.write(f'AUC: {auc}, precession: {prec}, Recall: {recall}\n')

        del tr


    matches = calc_accuracy(all_pred, all_targets)
    acc = matches.float().mean()
    all_pred = all_pred.numpy()
    all_targets = all_targets.numpy()

    prec, recall, auc = get_metrics(all_targets, all_pred)
    print(f'accuray: {acc}, AUC: {auc}, precession: {prec}, Recall: {recall}')
    f.write(f'accuray: {acc}, AUC: {auc}, precession: {prec}, Recall: {recall}')
    result = {'all_pred': all_pred, 'all_targets': all_targets}
    T.save(result, path.join('results',f'{name}_result'))
    f.close()



