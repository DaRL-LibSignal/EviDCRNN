from re import U
import torch
import numpy as np
import math
import torch.nn.functional as F
import scipy
import scipy.stats
from scipy.stats import t, norm, gamma

#def quantile_loss(y_pred, y_true):

def edl_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
    """
    mask = (y_true != 0).float()
    mask /= mask.mean()
    quantiles = [0.025, 0.5, 0.975]
    losses = []
    for i, q in enumerate(quantiles):
        errors =  y_true - torch.unbind(y_pred,3)[i]
        errors = errors * mask
        errors[errors != errors] = 0
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    
    print(gamma.size())
    print(nu.size())
    print(alpha.size())
    print(beta.size())
    print("target.size = ",target.size())
    """
    
    pi = torch.tensor(np.pi)
        
    x1 = torch.log(pi/nu)*0.5
    x2 = -alpha*torch.log(2.*beta*(1.+ nu))
    x3 = (alpha + 0.5)*torch.log( nu*(target - gamma)**2 + 2.*beta*(1. + nu) )
    x4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    """
    if self.reduction == 'mean': 
        return (x1 + x2 + x3 + x4).mean()
    elif self.reduction == 'sum':
        return (x1 + x2 + x3 + x4).sum()
    else:
        return (x1 + x2 + x3 + x4)
    """
    loss_value =  torch.abs(target - gamma)*(2*nu + alpha) * 0.1 #set default parameters as 0.1 
    """
    if self.reduction == 'mean': 
            return (loss_value).mean()
        elif self.reduction == 'sum':
            return (loss_value).sum()
        else:
            return (loss_value)
    """
    mse = (gamma-target)**2
    c = get_mse_coef(gamma, nu, alpha, beta, target).detach()
    modified_mse = mse*c

    loss = (x1 + x2 + x3 + x4).mean() + loss_value.mean() + modified_mse.mean()
    
    return loss
def get_mse_coef(gamma, nu, alpha, beta, y):
    """
    Return the coefficient of the MSE loss for each prediction.
    By assigning the coefficient to each MSE value, it clips the gradient of the MSE
    based on the threshold values U_nu, U_alpha, which are calculated by check_mse_efficiency_* functions.

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        y ([FloatTensor]): true labels.

    Returns:
        [FloatTensor]: [0.0-1.0], the coefficient of the MSE for each prediction.
    """
    alpha_eff = check_mse_efficiency_alpha(gamma, nu, alpha, beta, y)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta, y)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt()/delta).detach()
    return torch.clip(c, min=False, max=1.)

def check_mse_efficiency_alpha(gamma, nu, alpha, beta, y, reduction='mean'):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial alpha(numpy.array) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    
    """
    delta = (y-gamma)**2
    right = (torch.exp((torch.digamma(alpha+0.5)-torch.digamma(alpha))) - 1)*2*beta*(1+nu) / nu

    return (right).detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta, y):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial nu(torch.Tensor) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu+1)/nu
    return (beta*nu_1/alpha)

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

def student_t_mis(self,gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:

    mu = gamma
    sigma_sqr = beta / (nu*(alpha - 1))

    l = mu - sigma_sqr.sqrt()*1.96
    u = mu + sigma_sqr.sqrt()*1.96
    l = self.standard_scaler.inverse_transform(l)
    u = self.standard_scaler.inverse_transform(u)
    """
    pa1 = 2*alpha
    pa2 = gamma 
    pa3 = torch.sqrt((beta*(1+nu)/(nu*alpha)))   #Represent three parameters of student-t distribution

    h = pa3 * (torch.tensor(t.ppf(0.975, pa1.cpu())).cuda())
    l = pa2 - h
    u = pa2 + h
    mis_loss = u-l
    #second_part = -0.025*u + 0.025*l
    """
    """
    lp = torch.zeros(gamma.size()).cuda()
    up = torch.zeros(gamma.size()).cuda()

    confidence = 0.97625
    for i in range(10):
        h = pa3 * (torch.tensor(t.ppf(confidence, pa1.cpu())).cuda())
        l = pa2 - h
        u = pa2 + h
        confidence += 0.0025
        #print("lp.size = ",lp.size())
        #print("l.size = ",l.size())
        lp = lp + 0.1*l
        up = up + 0.1*u
    """
    rou=0.05
    rou=2./rou
    #second_part = second_part + 0.025*(up - lp)

    mis_loss = (u-l) + rou * torch.max(y_true - u , u-u) + rou * torch.max(l - y_true , u-u)
    return mis_loss.mean()

def width(self,gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:

    #mask = (y_true != 0).float()
    #mask /= mask.mean()
    """
    pa1 = 2*alpha
    pa2 = gamma 
    pa3 = torch.sqrt((beta*(1+nu)/(nu*alpha)))   #Represent three parameters of student-t distribution

    h = pa3 * (torch.tensor(t.ppf(0.975, pa1.cpu())).cuda())
    l = pa2 - h
    r = pa2 + h
    interval_width = (r-l)
    
    
    
    
    
    mu = gamma
    sigma_sqr = beta / (alpha - 1)

    """
    mu = gamma
    sigma_sqr = beta / (nu*(alpha - 1))
    l = mu - sigma_sqr.sqrt()*1.96
    u = mu + sigma_sqr.sqrt()*1.96
    l = self.standard_scaler.inverse_transform(l)
    u = self.standard_scaler.inverse_transform(u)
    
    return (u-l).mean()

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
