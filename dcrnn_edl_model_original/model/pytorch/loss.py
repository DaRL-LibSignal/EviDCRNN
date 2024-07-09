from re import U
import torch
import numpy as np
import math
import torch.nn.functional as F
import scipy
import scipy.stats
from scipy.stats import t, norm, gamma
import torch.optim as optim
#def quantile_loss(y_pred, y_true):

def edl_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor, lambda_coef) -> torch.Tensor:
    
    pi = torch.tensor(np.pi)
        
    x1 = torch.log(pi/nu)*0.5
    x2 = -alpha*torch.log(2.*beta*(1.+ nu))
    x3 = (alpha + 0.5)*torch.log( nu*(target - gamma)**2 + 2.*beta*(1. + nu) )
    x4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    loss_reg =  torch.abs(target - gamma)*(2*nu + alpha) #set default parameters as 0.1 
    loss = (x1 + x2 + x3 + x4).mean() + (lambda_coef * loss_reg).mean()    
    
    return loss,loss_reg

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = (y_pred  - y_true) ** 2
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def student_t_mis(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:

    
    pa1 = 2*alpha
    pa2 = gamma 
    pa3 = torch.sqrt((beta*(1+nu)/(nu*alpha)))   #Represent three parameters of student-t distribution

    h = pa3 * (torch.tensor(t.ppf(0.975, pa1.cpu())).cuda())
    l = pa2 - h
    u = pa2 + h
    mis_loss = u-l

    lp = torch.zeros(gamma.size()).cuda()
    up = torch.zeros(gamma.size()).cuda()

    confidence = 0.97625
    for i in range(10):
        h = pa3 * (torch.tensor(t.ppf(confidence, pa1.cpu())).cuda())
        l = pa2 - h
        u = pa2 + h
        confidence += 0.0025
        lp = lp + 0.1*l
        up = up + 0.1*u
    

    rou=0.05
    rou=2./rou

    second_part = -0.025*u + 0.025*l
    second_part = second_part + 0.025*(up - lp)

    mis_loss = mis_loss + rou*second_part

    return mis_loss.mean()

def width(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:

    #mask = (y_true != 0).float()
    #mask /= mask.mean()
    
    pa1 = 2*alpha
    pa2 = gamma 
    pa3 = torch.sqrt((beta*(1+nu)/(nu*alpha)))   #Represent three parameters of student-t distribution

    h = pa3 * (torch.tensor(t.ppf(0.975, pa1.cpu())).cuda())
    l = pa2 - h
    r = pa2 + h
    interval_width = (r-l)
    return interval_width.mean()

def confidence_interval(mu, std, sample_num, confidence=0.95):
    """
    Calculate confidence interval from mean and std for each predictions
    under the empricial t-distribution.
    
    If the sample_num is given as the 0, it will compute Gaussian confidence interval,
    not t-distribution
    
    If we use evidential network with following outputs
    : gamma, nu, alpha, beta = model(X)

    then we the std and mu will be
    : mu, std, freedom = gamma, (beta*(nu + 1))/(alpha*nu), 2*alpha
    : low_interval, up_interval = confidence_interval(mu, std, freedom)

    Args:
        mu(np.array): Numpy array of the predictive mean
        std(np.array): Numpy array of the predictive standard derivation
        sample_num(np.array or Int): Degree of freedom. It can be a [np.array] having same dimension
            with the mu and std.
        confidence(Float): confidence level, [0,1] (eg: 0.99, 0.95, 0.1)
    Return:
        low_interval(np.array), up_interval(np.array): confidence intervals
    """
    mu = mu
    std = std
    sample_num = sample_num

    n = sample_num
    h = std * (torch.tensor(t.ppf((1 + confidence) / 2, n.cpu())).cuda())
    low_interval = mu - h
    up_interval = mu + h
    return low_interval, up_interval

def ECE_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:
    
    mu = gamma
    std = torch.sqrt((beta*(1+nu))/(nu*alpha))
    Y = y_true
    sample_num = 2*alpha
    confidences = np.linspace(0, 1, num=100)
    calibration_errors = []
    interval_acc_list = []
    for confidence in confidences:
        low_interval, up_interval = confidence_interval(mu, std, sample_num, confidence=confidence)
        
        hit = 0
        a = (low_interval <= Y).float()
        b = (Y <= up_interval).float()
        c = a*b

        hit = c.sum()
        
        interval_acc = hit.item()/a.numel()
        interval_acc_list.append(interval_acc)
        calibration_errors.append((confidence - interval_acc)**2)

    return np.mean(np.sqrt(calibration_errors))

def Gaussian_distribution_mis(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:

    mask = (y_true != 0).float()
    mask /= mask.mean()
    
    mu = gamma
    #mu = mu.unsqueeze(-1)

    sigam = beta / (alpha - 1)
    sigam = torch.pow(sigam , 0.5)
    #sigam = sigam.unsqueeze(-1)

    rou = 0.05


    lx = (-1.96) * sigam + mu
    ux = 1.96 * sigam + mu

    ans_mis = ux - lx
    ans_mis = ans_mis + 4 / rou * (sigam / (math.sqrt(2* math.pi)) * torch.pow(math.e,(-torch.pow(ux - mu,2) / (2 * torch.pow(sigam , 2)))))

    ans_mis = ans_mis * mask
    ans_mis[ans_mis != ans_mis] = 0

    losses = []
    losses.append(ans_mis.unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    
    return loss