import torch
import numpy as np
import scipy.signal

# test sequence 1: target
target = torch.load('/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq1.pt', map_location=torch.device('cpu'))
print(target.shape)
# test sequence 1: prediction
prediction = torch.load('/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq1.pt', map_location=torch.device('cpu'))
print(prediction.shape)


c = np.corrcoef(target.squeeze(), prediction.squeeze(), rowvar = False).mean(axis = 0)
print(c)
print(c.shape)

reduce_dims = (0,1)

product = torch.sum(prediction * target, dim=reduce_dims) 
true = torch.sum(target, dim=reduce_dims)    
true_squared = torch.sum(torch.square(target), dim=reduce_dims)  
pred = torch.sum(prediction, dim=reduce_dims) 
pred_squared = torch.sum(torch.square(prediction), dim=reduce_dims) 
count = torch.sum(torch.ones_like(target), dim=reduce_dims) 

true_mean = true / count
pred_mean = pred / count

covariance = (product
            - true_mean * pred
            - pred_mean * true
            + count * true_mean * pred_mean)

true_var = true_squared - count * torch.square(true_mean)
pred_var = pred_squared - count * torch.square(pred_mean)

tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
correlation = covariance / tp_var

# print(correlation)
# print((correlation.mean()))

# d = np.correlate(a = prediction.squeeze(), v = target.squeeze())
# print(d)
# print(d.shape)

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

d = corr2_coeff(prediction.squeeze(), target.squeeze())
print(d)
print(d.shape)

e = scipy.signal.correlate2d(prediction.squeeze(), target.squeeze())
print(e)
print(e.shape)






