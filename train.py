import os
import argparse
import time
import random
from numpy.lib.function_base import average

from tqdm import tqdm

from collections import OrderedDict, namedtuple
from itertools import product

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score

import matplotlib.pyplot as plt

from dset import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--which',
                    help='Which model to train: 0-SVM, 1-KNN, 2-DANN, 3-LSTM',
                    default='012',
                    type=str)
parser.add_argument('--datapath',
                    help='The path to your SEED-IV',
                    default='/home/PublicDir/huhaoyi/seed_partition/dataset/seed_iv/eeg',
                    type=str)
parser.add_argument('--maxepochs',
                    help='Maximum training epochs',
                    default=1000,
                    type=int)
parser.add_argument('--lr',
                    help='Learning rate',
                    default='[1e-2]',
                    type=str)

# SVM hyper-parameters

# DANN hyper-parameters
parser.add_argument('--lambda_',
                    help='Lambda',
                    default='[1.0]',
                    type=str)
parser.add_argument('--alpha',
                    help='Coefficient of total loss',
                    default='[1.0]',
                    type=str)

# LSTM hyper-parameters
parser.add_argument('--hidden_size',
                    help='Hidden size of the LSTM, enter multiple values like [32,64,128] starts hyperparameter test',
                    default='[128]',
                    type=str)
parser.add_argument('--num_layers',
                    help='Number of LSTM layers',
                    default='[3]',
                    type=str)


args = parser.parse_args()


class BaselinesTrainApp:
    def __init__(self, dset_train, dset_test):
        self.x_train = dset_train.data.numpy().reshape((800, -1))
        self.y_train = dset_train.label.numpy()

        self.x_test = dset_test.data.numpy().reshape((100, -1))
        self.y_test = dset_test.label.numpy()

    def main(self):
        x_train = StandardScaler().fit_transform(self.x_train)
        x_test = StandardScaler().fit_transform(self.x_test)
        model_names = ['SVM', 'KNN']
        models = [SVC(kernel="linear", probability=False, class_weight='balanced'), KNeighborsClassifier(n_neighbors=30)]

        for i in range(len(model_names)):
            model_name = model_names[i]
            model = models[i]
            print('=' * 25, 'Model: {}'.format(model_name))
            start = time.time()
            model.fit(x_train, self.y_train)
            end = time.time()
            dur = end - start
            print('Train time: {:4d}min {:2d}sec'.format(int(dur // 60), int(dur % 60)))

            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            train_pred_accuracy = accuracy_score(self.y_train, train_pred)
            test_pred_accuracy = accuracy_score(self.y_test, test_pred)
            print('train_acc: {}\ntest_acc: {}'.format(train_pred_accuracy, test_pred_accuracy))

            f1_train = f1_score(self.y_train, train_pred, average=None)
            f1_test = f1_score(self.y_test, test_pred, average=None)
            print('train_f1: {}\ntest_f1: {}'.format(str(f1_train), str(f1_test)))

            train_cm_display = plot_confusion_matrix(model, x_train, self.y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
            train_cm_display.figure_.suptitle('{}: Train set confusion matrix'.format(model_name))
            train_cm_display.figure_.savefig('./figs/{}_train_cm.png'.format(model_name))

            test_cm_display = plot_confusion_matrix(model, x_test, self.y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
            test_cm_display.figure_.suptitle('{}: Test set confusion matrix'.format(model_name))
            test_cm_display.figure_.savefig('./figs/{}_test_cm.png'.format(model_name))


class DANNTrainApp:
    def __init__(self):
        self.dpath = args.datapath
        self.nworkers = os.cpu_count()
        self.maxepochs = args.maxepochs
        self.lr = eval(args.lr)
        self.lambda_ = eval(args.lambda_)
        self.alpha = eval(args.alpha)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    
    def get_runs(self):
        params = OrderedDict(lr = self.lr,
                             lambda_ = self.lambda_,
                             alpha = self.alpha)
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    def getModel(self, input_dim, output_dim, lambda_):
        dann = DANN(input_dim, output_dim)
        domain_clf = DomainClassifier(lambda_)
        if self.use_cuda:
            print('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            # if torch.cuda.device_count() > 1:
            #     dann = nn.DataParallel(dann)
            #     domain_clf = nn.DataParallel(domain_clf)
        else:
            print('Using cpu.')

        return dann.to(self.device), domain_clf.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def getDataset(self, feature, smooth_method, frq_bands, target_subject):
        source_dset = SEED_IV(self.dpath, feature, smooth_method, frq_bands, [i for i in range(1, 16) if i != target_subject], DANN='source')
        target_dset = SEED_IV(self.dpath, feature, smooth_method, frq_bands, [target_subject], DANN='target')

        self.batch_size = 128
        # if self.use_cuda:
        #     self.batch_size *= torch.cuda.device_count()

        # the size of source and target in each batch should be equal, otherwise the model tends to focus on source
        source_dloader = DataLoader(source_dset, batch_size=self.batch_size // 2, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True)
        target_dloader = DataLoader(target_dset, batch_size=self.batch_size // 2, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True)
        return source_dloader, target_dloader, source_dset.data.size()[1]

    def main(self):
        print('#' * 100)
        print('DANN\n')

        conditions = OrderedDict(features = ['de', 'psd'],
                                 smooth_methods = ['movingAve', 'LDS'],
                                 frq_bands = [['delta'], ['theta'], ['alpha'], ['beta'], ['gamma'], ['delta', 'theta', 'alpha', 'beta', 'gamma']])
        exps = [v for v in conditions.values()]
        
        runs = self.get_runs()
        for run in runs:
            for feature, smooth_method, frq_bands in product(*exps):
                comment = ' DANN lr={} feature={} smooth_method={} frq_bands={} lambda_={} alpha={}'.format(run.lr, feature, smooth_method, str(frq_bands), run.lambda_, run.alpha)
                dann_writer = SummaryWriter(comment=comment)

                print(comment + ' >>>')
                test_accuracies = []
                start_outer = time.time()
                for target_subject in range(1, 16):
                    print('Target on {}...'.format(target_subject))
                    source_dloader, target_dloader, input_dim = self.getDataset(feature, smooth_method, frq_bands, target_subject)
                    nbatches = min(len(source_dloader), len(target_dloader))

                    dann, domain_classifier = self.getModel(input_dim, 4, run.lambda_)
                    # weight initialization
                    dann.feature_extractor.apply(self.init_weights)
                    dann.emotion_classifier.apply(self.init_weights)
                    domain_classifier.apply(self.init_weights)

                    # optim = torch.optim.Adam(list(dann.parameters()) + list(domain_classifier.parameters()))
                    optim = torch.optim.SGD(list(dann.parameters()) + list(domain_classifier.parameters()), lr=run.lr)
                    
                    for epoch in tqdm(range(1, self.maxepochs+1)):
                        loss, emotion_loss, domain_loss = 0, 0, 0
                        emotion_pred_accuracy, domain_pred_accuracy = 0, 0

                        for (src_data, src_emotion_label, src_domain_label), (tgt_data, _, tgt_domain_label) in zip(source_dloader, target_dloader):
                            x = torch.cat([src_data, tgt_data], 0).to(self.device)
                            y_emotion = src_emotion_label.to(self.device)
                            y_domain = torch.cat([src_domain_label, tgt_domain_label], 0).to(self.device)

                            features = dann.feature_extractor(x)
                            pred_emotion = dann.emotion_classifier(features[:src_data.shape[0]])
                            pred_domain = domain_classifier(features)

                            # for nn.xxx, need to declare and then use because it is class
                            # for nn.functional.xxx, use directly
                            emotion_loss_b = F.nll_loss(pred_emotion, y_emotion)
                            emotion_loss += emotion_loss_b
                            domain_loss_b = F.nll_loss(pred_domain, y_domain)
                            domain_loss += domain_loss_b
                            loss_b = emotion_loss_b + run.alpha * domain_loss_b
                            loss += loss_b

                            optim.zero_grad()
                            loss_b.backward()
                            optim.step()

                            emotion_pred_accuracy += (pred_emotion.max(dim=1)[1] == y_emotion).float().mean().item()
                            domain_pred_accuracy += (pred_domain.max(dim=1)[1] == y_domain).float().mean().item()

                        epa_mean = emotion_pred_accuracy / nbatches  # expected to increase
                        dpa_mean = domain_pred_accuracy / nbatches  # expected to approximate 50%

                        dann_writer.add_scalar('loss/{}'.format(target_subject), loss, epoch)
                        dann_writer.add_scalar('emotion_loss/{}'.format(target_subject), emotion_loss, epoch)
                        dann_writer.add_scalar('domain_loss/{}'.format(target_subject), domain_loss, epoch)
                        dann_writer.add_scalar('emotion_pred_accuracy/{}'.format(target_subject), epa_mean, epoch)
                        dann_writer.add_scalar('domain_pred_accuracy/{}'.format(target_subject), dpa_mean, epoch)
                        dann_writer.flush()
                        dann_writer.close()

                        # if epoch == 1 or epoch % 100 == 0:
                        #     tqdm.write('Epoch {:6d}: emotion_pred_accuracy {:10.4f}, domain_pred_accuracy {:10.4f}'.format(epoch, epa_mean, dpa_mean))
                    
                    # validation
                    with torch.no_grad():
                        correct, total = 0, 0
                        for tx, ty_e, _ in target_dloader:
                            tx = tx.to(self.device, non_blocking=True)
                            ty_e = ty_e.to(self.device, non_blocking=True)

                            tpred_e = dann(tx)

                            correct += (tpred_e.max(dim=1)[1] == ty_e).float().sum().item()
                            total += tx.shape[0]
                        
                        acc = correct / total
                        print('Model subj{:d} transfer prediction accuracy: {:10.4f}'.format(target_subject, acc))
                        test_accuracies.append(acc)
                
                end_outer = time.time()
                dur_outer = end_outer - start_outer
                print('Train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))

                cv_acc_mean = torch.tensor(test_accuracies).mean().item()
                cv_acc_std = torch.tensor(test_accuracies).std().item()
                print('Leave-one-out cross validation, mean: {:10.4f}, std: {:10.4f}'.format(cv_acc_mean, cv_acc_std))


class LSTMTrainApp:
    def __init__(self):
        self.dir_path = args.datapath
        self.nworkers = os.cpu_count()
        self.lr = args.lr
        self.maxepochs = args.maxepochs
        
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_classes = 4

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

    def getModel(self, seq_len, input_size):
        model = LSTM_Classification(seq_len, input_size, self.hidden_size, self.num_layers, self.num_classes)

        if self.use_cuda:
            print('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            # if torch.cuda.device_count() > 1:
            #     dann = nn.DataParallel(dann)
            #     domain_clf = nn.DataParallel(domain_clf)
        else:
            print('Using cpu.')
        return model.to(self.device)
    
    def getDataset(self, feature, smooth_method, frq_bands, target_subject):
        dset_train = SEED_IV(self.dir_path, feature, smooth_method, frq_bands, [i for i in range(1, 16) if i != target_subject])
        dset_test = SEED_IV(self.dir_path, feature, smooth_method, frq_bands, [target_subject])

        dloader_train = DataLoader(dset_train, batch_size=128, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True)
        dloader_test = DataLoader(dset_test, batch_size=128, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True)
        return dloader_train, dloader_test, dset_train.data.size()[1]

    def main(self):
        print('#'*100)
        print('LSTM\n')

        for feature in ['de', 'psd']:
            for smooth_method in ['movingAve', 'LDS']:
                for frq_bands in [['delta'], ['theta'], ['alpha'], ['beta'], ['gamma'], ['delta', 'theta', 'alpha', 'beta', 'gamma']]:
                    print('\nFeature in use: {}\nFrequency bands in use: {} >>>'.format(feature+'_'+smooth_method, str(frq_bands)))
                    test_accuracies = []
                    start_outer = time.time()
                    for target_subject in range(1, 16):
                        print('Target on {}'.format(target_subject))
                        dloader_train, dloader_test, seq_dim = self.getDataset(feature, smooth_method, frq_bands, target_subject)
                        
                        seq_len = 2
                        input_size = seq_dim // 2

                        model = self.getModel(seq_len, input_size)
                        # optim = opt.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.1)
                        optim = opt.Adam(model.parameters(), lr=self.lr)

                        for epoch in tqdm(range(1, self.maxepochs+1)):
                            train_pred_right = 0
                            train_total = 0
                            for x, y in dloader_train:
                                x = x.view(-1, seq_len, input_size).to(self.device)
                                y = y.to(self.device)

                                pred = model(x)

                                loss = F.nll_loss(pred, y)

                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                                train_pred_right += (pred.max(dim=1)[1] == y).float().sum().item()
                                train_total += x.shape[0]
                            
                            train_pred_acc = train_pred_right / train_total
                            if epoch == 1 or epoch % 100 == 0:
                                tqdm.write('Epoch {:6d}: train_pred_accuracy {:10.4f}'.format(epoch, train_pred_acc))
                        
                        with torch.no_grad():
                            test_pred_right = 0
                            test_total = 0
                            for x, y in dloader_test:
                                x = x.view(-1, seq_len, input_size).to(self.device)
                                y = y.to(self.device)

                                pred = model(x)

                                test_pred_right += (pred.max(dim=1)[1] == y).float().sum().item()
                                test_total += x.shape[0]
                            
                            test_pred_acc = test_pred_right / test_total
                        
                        test_accuracies.append(test_pred_acc)
                    
                    end_outer = time.time()
                    dur_outer = end_outer - start_outer
                    print('Train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))

                    cv_acc_mean = torch.tensor(test_accuracies).mean().item()
                    cv_acc_std = torch.tensor(test_accuracies).std().item()
                    print('Leave-one-out cross validation, mean: {:10.4f}, std: {:10.4f}'.format(cv_acc_mean, cv_acc_std))


if __name__ == '__main__':
    for feature in [None, 'psd', 'de']:
        if not feature == None:
            print('#'*50, ' Feature: {}'.format(feature))
            print()

            indices = list(range(900))
            random.shuffle(indices)
            train_indices = indices[:800]
            test_indices = indices[800:]
            dset_train = ArtDataset(feature=feature, choice=train_indices)
            dset_test = ArtDataset(feature=feature, choice=test_indices)

            if '0' in args.which:
                BaselinesTrainApp(dset_train, dset_test).main()
            
            if '1' in args.which:
                pass
            
            if '2' in args.which:
                DANNTrainApp().main()
            
            if '3' in args.which:
                LSTMTrainApp().main()
        else:
            for frq in ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']:
                pass
            
