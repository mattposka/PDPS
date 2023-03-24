import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.lovasz_losses import lovasz_softmax
import torch.nn.functional as F
from torchmetrics import JaccardIndex

import kornia as K
from kornia import morphology as morph

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, inputs, targets): # pred,label
        smooth = 1

        if inputs.dim()>2:
            #inputs = F.sigmoid(inputs)       
            inputs = torch.softmax(inputs,dim=1)
            inputs = inputs[:,1,:,:]

        #target_one_hot = torch.nn.functional.one_hot(targets)
        #target_one_hot = target_one_hot.transpose(1,3)

        #intersection = torch.sum(inputs*target_one_hot)
        inputs = inputs.flatten()
        targets = targets.flatten()
        intersection = torch.sum(inputs*targets)

        score = 2. * (intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)
        score = 1 - score
        return score

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target): # pred,label
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if inputs.dim()>2:
            #inputs = F.sigmoid(inputs)       
            inputs = torch.softmax(inputs,dim=1)
            inputs = inputs[:,1,:,:]

        #target_one_hot = torch.nn.functional.one_hot(targets)
        #target_one_hot = target_one_hot.transpose(1,3)

        dims = (1,2)
        
        intersection = torch.sum(inputs*targets, dims)
        fps = torch.sum(inputs*(1-targets), dims)
        fns = torch.sum((1-inputs)*targets, dims)
        #print('intersection :',intersection)
        #print('fps :',fps)
        #print('fns :',fns)

        numerator = intersection
        denominator = intersection + alpha*fps + beta * fns
       
        Tversky = (numerator + smooth) / (denominator + smooth)
        #print('tversky :',Tversky)
        #print('torch.mean(1.0-Tversky) :',torch.mean(1.0-Tversky))
        return torch.mean(1.0 - Tversky)

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=2.0):

        #comment out if your model contains a sigmoid or equivalent activation layer
        if inputs.dim()>2:
            #inputs = F.sigmoid(inputs)       
            inputs = torch.softmax(inputs,dim=1)
            inputs = inputs[:,1,:,:]
        
        #target_one_hot = torch.nn.functional.one_hot(targets)
        #target_one_hot = target_one_hot.transpose(1,3)

        dims = (1,2)
        
        intersection = torch.sum(inputs*targets, dims)
        fps = torch.sum(inputs*(1-targets), dims)
        fns = torch.sum((1-inputs)*targets, dims)

        numerator = intersection
        denominator = intersection + alpha*fps + beta * fns
       
        Tversky = (numerator + smooth) / (denominator + smooth)
        FocalTversky = (torch.mean(1 - Tversky))**gamma
                       
        return FocalTversky

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        if inputs.dim()>2:
            #inputs = F.sigmoid(inputs)       
            inputs = torch.softmax(inputs,dim=1)
            inputs = inputs[:,1,:,:]
        
        #target_one_hot = torch.nn.functional.one_hot(targets)
        #target_one_hot = target_one_hot.transpose(1,3)

        dims = (1,2)
        intersection = torch.sum(inputs*targets, dims)
        
        ##intersection is equivalent to True Positive count
        ##union is the mutually inclusive area of all labels & predictions 
        #intersection = (inputs * targets).sum()
        union = torch.sum(inputs + targets - (inputs * targets),dims)
        
        IoU = (intersection + smooth)/(union + smooth)
        return torch.mean(1 - IoU)

#class LovaszHingeLoss(nn.Module):
#    def __init__(self, weight=None, size_average=True):
#        super(LovaszHingeLoss, self).__init__()
#
#    def forward(self, inputs, targets):
#
#        inputs = torch.softmax(inputs,axis=1)[:,1,:,:]
#        b,h,w = inputs.shape
#        inputs = inputs.view(b,1,h,w)
#        Lovasz = lovasz_softmax(inputs, targets, per_image=False)                       
#        return Lovasz

def boundary_map(pred, one_hot_gt, theta, theta0):
    # boundary map
    gt_b = F.max_pool2d(
        1 - one_hot_gt, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    gt_b -= 1 - one_hot_gt
    pred_b = F.max_pool2d(
        1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    pred_b -= 1 - pred

    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)
    return gt_b_ext, pred_b_ext, gt_b, pred_b

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    after: https://github.com/yiskw713/boundary_loss_for_remote_sensing
    'To better account for boundary pixels we select BF1 metric (see original work [9] and its extension [12])
    to construct a differentiable surrogate and use it in training.
    The surrogate is not used alone for training, but as a weighted sum with IoU loss (from directIoUoptimization).
    We found that the impact of the boundary component of the loss function should be gradually increased during training,
    and so we proposed a policy for the weight update.'
    'Via the trial and error process we set θ0 to 3 and θ to 5-7 as a proper choice,
    because theses values deliver the most accurate boundaries in all experiments.'
    'for LBF1,IoUloss, it requires an additional procedure for mini grid-search:
    after the 8th epoch for every 30 epochs and for every weight w∈{0.1,0.3,0.5,0.7,0.9}
    in equation (BCE+wLBF1+ (1−w)LIoU) a network was trained.
    Then the best weight is chosen, and the process repeats'
    """

    def __init__(self, apply_nonlin='Sigmoid', theta0=3, theta=5, precision='half', debug=False):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.apply_nonlin = apply_nonlin
        self.precision = precision
        self.debug=debug
        self.epsilon = 1e-7
        #for debug only
        self.idx=0

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bath
        """

        n, c, h, w = pred.shape

        pred = torch.softmax(pred,axis=1)
        y_onehot = torch.nn.functional.one_hot(gt)
        y_onehot = y_onehot.transpose(1,3)

        if self.precision=='half':
            y_onehot=y_onehot.half()

        gt_b_ext, pred_b_ext, gt_b, pred_b = boundary_map(pred, y_onehot, self.theta, self.theta0)
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision: TP/(TP+FP)
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + self.epsilon)
        # Recall: TP/(TP+FN)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + self.epsilon)

        # Boundary F1 Score ~ Dice
        BF1 = 2 * P * R / (P + R + self.epsilon)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors


def lovasz_hinge_flat(logits, labels, ignore_index):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore_index is not None:
        mask = labels != ignore_index
        logits = logits[mask]
        labels = labels[mask]
    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszLoss(nn.Module):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return lovasz_hinge_flat(logits, labels, self.ignore_index)

class FocalLoss(nn.Module):

    def __init__(self,alpha,gamma):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,alpha])).cuda() # to put on cuda or cpu
        self.gamma = 2.0

    def forward(self, inputs, targets):

        p = torch.softmax(inputs,axis=1)
        ce = self.ce(inputs, targets)
        loss = torch.pow(1 - p, self.gamma) * ce
        
        return torch.mean(loss)
        