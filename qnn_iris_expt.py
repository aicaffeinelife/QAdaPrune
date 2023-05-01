import pennylane as qml
import pennylane.numpy as qnp
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import count_nonzero
from qml_utils import load_data, mse_loss, accuracy
import argparse

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def ansatz(params, data):
    qubits = params.shape[1]
    qml.AngleEmbedding(data, wires=range(qubits), rotation="X")
    for p in params:
        layer(p, qubits)
    return qml.expval(qml.PauliZ(0))

def layer(param, wires):
    for i in range(wires):
        qml.Rot(*param[i], wires=i)
    qml.broadcast(qml.CZ, pattern='chain', wires=range(wires))



def optimize_and_prune(opt, cost_fn, grad_fn, ckt, iparams, 
                       wires=4, win_sz=5, steps=100, tol=0.001, **kwargs):
    # params = qnp.random.uniform(-np.pi, np.pi, size=param_shape, requires_grad=True)
    params = iparams.copy()
    params.requires_grad = True
    param_shape = params.shape
    print("Param shape: ", param_shape)
    X, y = kwargs['data'], kwargs['label']
    grad_pts = kwargs['grad_pts'] if 'grad_pts' in kwargs else 1
    grad_buffer = qnp.zeros_like(params, requires_grad=False)
    # opt = qml.GradientDescentOptimizer(stepsize=0.4)
    igrad = grad_fn(params, X[:grad_pts])
    print("Igrad shape: ", igrad.shape)
    grad_buffer = np.abs(grad_buffer - igrad)
    tau = np.ones_like(np.prod(param_shape))*1/np.prod(param_shape)
    
    cost_hists = []
    acc_hists = []
    idx = None
    for i in range(steps):
        out, new_cost = opt.step_and_cost(cost_fn, ckt, params, (X, y))
        params = out[1]
        if idx is not None:
            params = params.flatten()
            params[idx] = 0 # apply pruning learned previously  
            params = params.reshape(param_shape)
            
        ngrad = grad_fn(params, X[:grad_pts])
        tau = tau - tau * softmax(ngrad.flatten()) # update threshold
        grad_buffer = np.abs(grad_buffer - ngrad)
        cost_hists.append(new_cost)
        acc = accuracy(ckt, params, (X,y))
        acc_hists.append(acc)
        print(f"Step: {i} | Cost: {new_cost}")
        if i!= 0 and i % win_sz == 0:
            idx = build_saliency(grad_buffer, tau)
            print(f"Dropping {idx} for threshold {tau}")
            # print("Building saliency....")
            # idx, thresh = build_saliency(grad_buffer, kappa)
            # print(f"Dropping: {idx} for threshold: {thresh}")
            tmp = params.flatten()
            tmp[idx] = 0
            params = tmp.reshape(param_shape)
            grad_buffer = np.zeros_like(params) # experimental
            
        if new_cost < tol:
            break
    # print("Params after convergence: ", params)
    sparsity = get_sparsity(params)
    print("Param sparsity after training: ", sparsity)
    return cost_hists, acc_hists, sparsity



def build_saliency(grad_buffer, thresh_vec):
    gbuf = grad_buffer.flatten()
    prune_idx = []
    for i, g in enumerate(gbuf):
        if g < thresh_vec[i]:
            prune_idx.append(i)
    return prune_idx


def get_sparsity(arr):
    return 1.0 - count_nonzero(arr)/arr.size






def optimize(opt, cost_fn, ckt, iparams, data, steps=100, tol=0.001):
    params = iparams.copy()
    params.requires_grad = True
    cost_hists, acc_hists = [] , []
    icost = cost_fn(ckt, params, data)
    cost_hists.append(icost)
    # prev_cost = icost
    for i in range(steps):
        print("Param shape: ", params.shape)
        out, new_cost = opt.step_and_cost(cost_fn, ckt, params, data)
        params = out[1]
        cost_hists.append(new_cost)
        acc = accuracy(ckt, params, data)
        acc_hists.append(acc)
        # conv = np.abs(new_cost) - prev_cost
        if i % 2 == 0:
            print(f"Step: {i} | Cost: {new_cost}")
        if new_cost < tol:
            break
        prev_cost = new_cost
    return cost_hists, acc_hists


def _loss_mse(label, pred):
    loss = 0.0
    for l, p in zip(label, pred):
        mse = (l - p)**2
        loss += mse
    return loss / len(label)

def cost_fn(qckt, params, data):
    """Thin wrapper around cost_mse

    Args:
        qckt (qml.QNode): Quantum circuit with fmt (params, data)
        params (torch.tensor/np.ndarray): Params of ckt
        data (tuple): Dataset

    Returns:
        loss(torch.tensor/np.array): loss over dataset
    """
    pred = [qckt(params, x) for x in data[0]]
    # print(pred)
    loss = _loss_mse(data[1], pred)
    return loss


def load_init_params(param_file):
    init_params = pickle.load(open(param_file, 'rb'))
    return init_params
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QNN training with pruning')    
    parser.add_argument('-s', '--save', type=str, default='iris_qnn', help='Save prefix')
    
    args = parser.parse_args()
    
    
    dev = qml.device('default.qubit', wires=4, shots=10000)
    qckt = qml.QNode(ansatz, device=dev, interface='autograd')
    
    X,y = load_data('../QML_MetaOPT/data/iris_classes1and2_scaled.txt')
    
    layers = list(range(4, 9))
    hist_layer = {}
    hist_np_layer = {}
    init_params = []
    
    print("Running expt with pruning....")
    grad_fn = qml.grad(qckt, argnum=0)
    for i, l in enumerate(layers):
        print("~"*20, f"Layer -> {l}", "~"*20)
        optim = qml.RMSPropOptimizer(stepsize=0.1)
        lparams = qnp.random.uniform(-np.pi, np.pi, size=(l, 4, 3), requires_grad=False)
        init_params.append(lparams)
        _hist = optimize_and_prune(optim, cost_fn, grad_fn, qckt, lparams, data=X, 
                            label=y, grad_pts=1, tol=0.01, steps=50)
        hist_layer[l] = _hist
    
    
    print("Running expt without pruning...")
    for i, l in enumerate(layers):
        print("~"*20, f"Layer -> {l}", "~"*20)
        optim = qml.RMSPropOptimizer(stepsize=0.1)
        params_np = init_params[i]
        _hist_np = optimize(optim, cost_fn, qckt, params_np, data=(X,y), steps=50)
        hist_np_layer[l] = _hist_np
    
    with open(f'{args.save}_pruned.pkl', 'wb') as ofnp:
        pickle.dump(hist_layer, ofnp)
    
    with open(f'{args.save}_unpruned.pkl', 'wb') as ofp:
        pickle.dump(hist_np_layer, ofp)
    
    # with open('iris_qnn_unpruned.pkl', 'wb') as ofile:
    #     pickle.dump(hist_layer, ofile)

    
