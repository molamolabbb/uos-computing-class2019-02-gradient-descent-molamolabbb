#!/usr/bin/env python
# coding: utf-8

# In[59]:


def step(v, direction, step_size):
    return [ v[i] + step_size*direction[i] for i in range(len(v))]
    
def move_point_along_ei(x,i,h):
    e = [1 if i == j else 0 for j in range(len(x))]
    return step(x,e,float(h))

def partial_difference_quotient(f,v,i,h):
    return float((f(move_point_along_ei(v,i,h))-f(v))/h)

def estimate_gradient(f,v,h=0.00001):
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]
'''

def minimize(f,df,x0,step_sizes=[10,1,0.1,0.01,0.001],tol=1e-5):
    n=0
    X_n = x0
    df = estimate_gradient(f,X_n)
    while X_n:
        #X_n1 = X_n - gamma*estimate_gradient(f,X_n)
        mini = [abs(f(X_n - gamma*estimate_gradient(f,X_n))-f(X_n)) for gamma in step_sizes ]
        if min(mini) < tol :
            return X_n
            break
        X_n = X_n1


'''


# In[101]:


def minimize(f,df,x0,step_sizes=[10,1,0.1,0.01,0.001],tol=1e-5):
    X_n = x0
    while X_n:
        mini = [f(step(X_n,df(X_n),-1*i))-f(X_n) for i in step_sizes ] 
        if abs(min(mini)) < tol :
            return X_n
            #break
        for i in step_sizes:
            if abs(min(mini))==abs(f(step(X_n,df(X_n),-1*i))-f(X_n)):
                gamma = i
        X_n = step(X_n,df(X_n),-1*gamma)

