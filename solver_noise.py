from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *
from biodata import get_biodata
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from functools import partial



def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
    )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value


# Training settings
class Solver_Noise(object):
    def __init__(self, args, batch_size=64, source='renji',
                 target='huashan', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10, datafolder = '/root/autodl-tmp/', ys =[],yt=[]):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.datafolder = datafolder
        self.ys = ys
        self.yt = yt
        self.bestAcc1 = 0
        self.bestAcc2 = 0
        self.bestAcc3 = 0
        self.bestAcc11 = 0
        self.bestAcc21 = 0
        self.bestAcc31 = 0
        
        print('dataset loading')
        self.s_train_dataset, _ = get_biodata(source, self.batch_size, datafolder=self.datafolder)
        self.s_test_dataset, _ = get_biodata(source, self.batch_size,  datafolder=self.datafolder)
        self.t_train_dataset, _ = get_biodata(target, self.batch_size,  datafolder=self.datafolder)
        self.t_test_dataset, _ = get_biodata(target, self.batch_size,  datafolder=self.datafolder)
        print('load finished!')

        self.G = Generator()
        self.C1 = Classifier()
        self.C2 = Classifier()
        self.D = Classifier()
        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        #self.D.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(0)
        self.datazip = enumerate(zip(self.s_train_dataset,self.t_train_dataset))

        if self.target == 'huashan':
            datanum = 310
        elif self.target == 'renji':
            datanum = 610
        else:
            datanum = 634   

        if self.source  == 'huashan':
            datanum_s = 310
        elif self.source == 'renji':
            datanum_s = 610
        else:
            datanum_s = 634  
        new_ys = np.zeros([datanum_s, 2])#source
        new_yt = np.zeros([datanum, 2])#2 means classnum

        for batch_idx, ((img_s, label_s, index_s), (img_t, _, index_t)) in self.datazip:
            
            index_s = index_s.numpy()
            index_t = index_t.numpy()
            label_ss = label_s

            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            #to_cuda
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            label_s = label_s.long().cuda()
            label_s = Variable(label_s)
            #print(img_s.device)
            #print(img_t.device)
            #print(label_s.device)
            
            self.reset_grad()

            #通过特征提取器G，提取源域图片的特征
            feat_s = self.G(img_s)
            
            #通过两个分类器输出分类概率
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            logsoftmax = nn.LogSoftmax(dim=1).cuda()
            softmax = nn.Softmax(dim=1).cuda()

            #step A: 使用源域数据来训练一个初始模型，同时初始化ys_tilde(new_ys)
            if epoch < 20:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                onehot = torch.zeros(label_ss.size(0), 2).scatter_(1, label_ss.view(-1, 1), 10.0)
                onehot = onehot.numpy()
                new_ys[index_s, :] = onehot
                loss_s = loss_s1 + loss_s2
                loss_s.backward()
                self.opt_g.step()
                self.opt_c1.step()
                self.opt_c2.step()
                self.reset_grad()

                if batch_idx % self.interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data, loss_s2.data))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s\n' % (loss_s1.data, loss_s2.data))
                    record.close()
                continue 
                
            #Step B：
                #a：最大化分类器差异的同时，减小源域数据的兼容性损失，最小化源域和目标域的KL散度（改良版）。
                #b：最小化分类器差异
            elif epoch < 100 and epoch >= 20:
                #part a
                feat_s = self.G(img_s)
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                
                yy_s = self.ys
                yy_s = yy_s[index_s,:]
                yy_s = torch.FloatTensor(yy_s)
                yy_s = yy_s.cuda()
                yy_s = torch.autograd.Variable(yy_s,requires_grad = True)
                last_ys_var = softmax(yy_s)

                yy_t = self.yt
                yy_t = yy_t[index_t,:]
                yy_t = torch.FloatTensor(yy_t)
                yy_t = yy_t.cuda()
                yy_t = torch.autograd.Variable(yy_t,requires_grad = True)
                last_yt_var = softmax(yy_t)

                #计算损失
                loss_mmd = mmd_loss(feat_s, feat_t)
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s = loss_s1 + loss_s2 + loss_mmd

                #KL
                lc_s1 = torch.mean(softmax(output_s1)*(logsoftmax(output_s1)-torch.log((last_ys_var))))
                lc_s2 = torch.mean(softmax(output_s2)*(logsoftmax(output_s2)-torch.log((last_ys_var))))
                lc_s = (lc_s1+lc_s2)/2
                lo_s = criterion(last_ys_var, label_s)

                lc_t1 = torch.mean(softmax(output_t1)*(logsoftmax(output_t1)-torch.log((last_yt_var))))
                lc_t2 = torch.mean(softmax(output_t2)*(logsoftmax(output_t2)-torch.log((last_yt_var))))
                lc_t = (lc_t1+lc_t2)/2

                '''discrepancy'''
                loss_dis = self.discrepancy(output_t1, output_t2)

                loss = loss_s + lc_s +lo_s +lc_t - loss_dis
                loss.backward()
                self.opt_c1.step()
                self.opt_c2.step()
                self.reset_grad()

                #part b
                for i in range(self.num_k):
                    
                    feat_t = self.G(img_t)
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)

                    '''discrepancy'''
                    loss_dis = self.discrepancy(output_t1, output_t2)
                    
                    loss_dis.backward()
                    self.opt_g.step()
                    self.reset_grad()
                
                yy_s.data.sub_(600*yy_s.grad.data)
                new_ys[index_s,:] = yy_s.data.cpu().numpy()  
                yy_t.data.sub_(600*yy_t.grad.data)
                new_yt[index_t,:] = yy_t.data.cpu().numpy()   
            #step C:使用交叉熵进行最后的微调
            else:
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                yy_s = self.ys
                yy_s = yy_s[index_s,:]
                yy_s = torch.FloatTensor(yy_s)
                yy_s = yy_s.cuda()
                yy_s = torch.autograd.Variable(yy_s,requires_grad = True)
                last_ys_var = softmax(yy_s)

                yy_t = self.yt
                yy_t = yy_t[index_t,:]
                yy_t = torch.FloatTensor(yy_t)
                yy_t = yy_t.cuda()
                yy_t = torch.autograd.Variable(yy_t,requires_grad = True)
                last_yt_var = softmax(yy_t) 

                loss_s = criterion(output_s1, last_ys_var)
                loss_t = criterion(output_t1, last_yt_var)
                loss = loss_s+loss_t
                loss.backward()
                self.opt_g.step()
                self.opt_c1.step()
                self.opt_c2.step()
                self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data, loss_s2.data, loss_dis.data))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.data, loss_s1.data, loss_s2.data))
                    record.close()
        datafolder = self.datafolder
        if epoch < 100:
            # save y_tilde
            ys = new_ys
            ys_file = datafolder + "ys.npy"
            np.save(ys_file,ys)
            ys_record = datafolder + "record/ys_%03d.npy" % epoch
            np.save(ys_record,ys)

            yt = new_yt
            yt_file = datafolder + "yt.npy"
            np.save(yt_file,yt)
            yt_record = datafolder + "record/yt_%03d.npy" % epoch
            np.save(yt_record,yt)

        return batch_idx
    
    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        
        if epoch < 20:
            self.dataset_test = enumerate(zip(self.s_train_dataset,self.t_train_dataset))
            for batch_idx, ((img, label,_), (_, __,___)) in self.dataset_test:
                #print(img,label) 
                img, label = img.cuda(), label.long().cuda()
                img, label = Variable(img, volatile=True), Variable(label)
                feat = self.G(img)
                output1 = self.C1(feat)
                output2 = self.C2(feat)
                test_loss += F.nll_loss(output1, label).data
                output_ensemble = output1 + output2
                pred1 = output1.data.max(1)[1]
                pred2 = output2.data.max(1)[1]
                pred_ensemble = output_ensemble.data.max(1)[1]
                k = label.data.size()[0]
                correct1 += pred1.eq(label.data).cpu().sum()
                correct2 += pred2.eq(label.data).cpu().sum()
                correct3 += pred_ensemble.eq(label.data).cpu().sum()
                
                size += k
            test_loss = test_loss / size
            if self.bestAcc1 < correct1:
                self.bestAcc1 = correct1
            if self.bestAcc2 < correct2:
                self.bestAcc2 = correct2
            if self.bestAcc3 < correct3:
                self.bestAcc3 = correct3
            print(
                '\nSource test set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.2f}%) Accuracy C2: {}/{} ({:.2f}%) Accuracy Ensemble: {}/{} ({:.2f}%) \n'.format(
                    test_loss, self.bestAcc1, size,
                    100. * self.bestAcc1 / size, self.bestAcc2, size, 100. * self.bestAcc2 / size, self.bestAcc3, size, 100. * self.bestAcc3 / size))
        elif epoch < 100:
            self.dataset_test = enumerate(zip(self.s_test_dataset,self.t_test_dataset))
            pred_s1 = []
            pred_s2 = []
            pred_s3 = []
            label_s = []
            for batch_idx, ((_, __, ___), (img, label, _)) in self.dataset_test:
                #print(img,label)
                img, label = img.cuda(), label.long().cuda()
                img, label = Variable(img, volatile=True), Variable(label)
                feat = self.G(img)
                output1 = self.C1(feat)
                output2 = self.C2(feat)
                test_loss += F.nll_loss(output1, label).data
                output_ensemble = output1 + output2
                pred1 = output1.data.max(1)[1]
                pred2 = output2.data.max(1)[1]
                pred_ensemble = output_ensemble.data.max(1)[1]
                pred_s1.append(pred1)
                pred_s2.append(pred1)
                pred_s3.append(pred1)
                label_s.append(label.data)

                k = label.data.size()[0]
                correct1 += pred1.eq(label.data).cpu().sum()
                correct2 += pred2.eq(label.data).cpu().sum()
                correct3 += pred_ensemble.eq(label.data).cpu().sum()
                size += k
            test_loss = test_loss / size

            #classifier 1
            precision_1 = precision_score(np.array(pred_s1), np.array(label_s), average='micro')
            recall_1 = recall_score(np.array(pred_s1), np.array(label_s), average='micro')
            f1_1 = f1_score(np.array(pred_s1), np.array(label_s), average='micro')
            #classifier 2
            precision_2 = precision_score(np.array(pred_s2), np.array(label_s), average='micro')
            recall_2 = recall_score(np.array(pred_s2), np.array(label_s), average='micro')
            f1_2 = f1_score(np.array(pred_s2), np.array(label_s), average='micro')
            '''#essenble
            precision_3 = precision_score(np.array(pred_s3), np.array(label_s), average='micro')
            recall_3 = recall_score(np.array(pred_s3), np.array(label_s), average='micro')
            f1_3 = f1_score(np.array(pred_s3), np.array(label_s), average='micro')'''

            precision_3 = precision_1+precision_2
            precision_3 = precision_3 / 2
            if self.precision_3 <precision_3:
                self.precision_3 = precision_3
            
            recall_3 = recall_1+recall_2
            recall_3 = recall_3 / 2
            if self.recall_3 <recall_3:
                self.recall_3 = recall_3

            f1_3 = f1_1+f1_2
            f1_3 = f1_3 / 2
            if self.f1_3 <f1_3:
                self.f1_3 = f1_3

            correct3 = correct1+correct2
            correct3 = correct3 / 2
            if self.bestAcc31 < correct3:
                self.bestAcc31 = correct3
            '''print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.2f}%) Accuracy C2: {}/{} ({:.2f}%) Accuracy Ensemble: {}/{} ({:.2f}%) \n'.format(
                    test_loss, self.bestAcc1, size,
                    100. * self.bestAcc11 / size, self.bestAcc21, size, 100. * self.bestAcc21 / size, self.bestAcc31, size, 100. * self.bestAcc31 / size))'''
            print(
                '\nTest set: Average loss: {:.4f},  Accuracy Ensemble: {}/{} ({:.2f}%) Precision Ensemble: ({:.2f}%) Recall Ensemble:  ({:.2f}%) F1 Score Ensemble: ({:.2f}%)\n'.format(
                    test_loss, self.bestAcc31, size, 100. * self.bestAcc31 / size), self.precision_3, self.recall_3, self.f1_3)
            if save_model and epoch % self.save_epoch == 0:
                torch.save(self.G,
                        '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
                torch.save(self.C1,
                        '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
                torch.save(self.C2,
                        '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            if record_file:
                record = open(record_file, 'a')
                print('recording %s', record_file)
                record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
                record.close()
        else:
            self.dataset_test = enumerate(zip(self.s_test_dataset,self.t_test_dataset))
            for batch_idx, ((_, __, ___), (img, label, _)) in self.dataset_test:
                #print(img,label)
                img, label = img.cuda(), label.long().cuda()
                img, label = Variable(img, volatile=True), Variable(label)
                feat = self.G(img)
                output1 = self.C1(feat)
                output2 = self.C2(feat)
                test_loss += F.nll_loss(output1, label).data
                output_ensemble = output1 + output2
                pred1 = output1.data.max(1)[1]
                pred2 = output2.data.max(1)[1]
                pred_ensemble = output_ensemble.data.max(1)[1]
                k = label.data.size()[0]
                correct1 += pred1.eq(label.data).cpu().sum()
                correct2 += pred2.eq(label.data).cpu().sum()
                correct3 += pred_ensemble.eq(label.data).cpu().sum()
                size += k
            test_loss = test_loss / size
            if self.bestAcc11 < correct1:
                self.bestAcc11 = correct1
            if self.bestAcc21 < correct2:
                self.bestAcc21 = correct2
            if self.bestAcc31 < correct3:
                self.bestAcc31 = correct3
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.2f}%) Accuracy C2: {}/{} ({:.2f}%) Accuracy Ensemble: {}/{} ({:.2f}%) \n'.format(
                    test_loss, self.bestAcc1, size,
                    100. * self.bestAcc11 / size, self.bestAcc21, size, 100. * self.bestAcc21 / size, self.bestAcc31, size, 100. * self.bestAcc31 / size))
            if save_model and epoch % self.save_epoch == 0:
                torch.save(self.G,
                        '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
                torch.save(self.C1,
                        '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
                torch.save(self.C2,
                        '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            if record_file:
                record = open(record_file, 'a')
                print('recording %s', record_file)
                record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
                record.close()