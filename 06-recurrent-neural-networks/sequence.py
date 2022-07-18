# %matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt


# generate time data
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)

err = torch.normal(0, 0.2, (T,))
x = torch.sin(0.01 * time)  + err

d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# plt.show()


# generate train data
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
  features[:, i] = x[i : T - tau + i]
label = x[tau : T]

batch_size, n_train = 15, 600
train_iter = d2l.load_array((features[:n_train], label[:n_train]),batch_size, is_train=True)


# use MLP to train model
def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.xavier_uniform_(m.weight)

def get_net():
  net = nn.Sequential(nn.Linear(tau, 10), nn.ReLU(), nn.Linear(10, 1))
  net.apply(init_weights)
  return net

loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
  trainer = torch.optim.Adam(net.parameters(), lr)
  for epoch in range(epochs):
    for X, y in train_iter:
      trainer.zero_grad()
      out = net(X)
      y = y.reshape(out.shape)
      l = loss(out, y)
      l.sum().backward()
      trainer.step()
    print(f'epoch {epoch + 1}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)    


# obe-step-ahead prediction
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()],
         'time', 'x', legend=['data', '1-step preds'],
         xlim=[1, 1000], figsize=(6, 3))
# plt.show()         


# k-step-ahead prediction
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
  multistep_preds[i] = net(multistep_preds[i - tau : i])
d2l.plot([time, time[tau:], time[n_train+tau:]], 
         [x.detach().numpy(), onestep_preds.detach().numpy(), 
         multistep_preds[n_train+tau:].detach().numpy()],
         'time', 'x', legend=['data', '1-step preds', 'k-step preds'],
         xlim=[1, 1000], figsize=(6, 3))
# plt.show()        

max_steps = 64
features = torch.zeros((T - tau, tau + max_steps))
for i in range(tau):
  features[:, i] = x[i: T - tau + i]

for i in range(tau, tau + max_steps):
  print(features[:, i].shape, net(features[:, i-tau:i]).reshape(-1).shape)
  features[:, i] = net(features[:, i-tau: i]).reshape(-1)

steps = [1, 4, 16, 64]
d2l.plot([time[tau:]] * len(steps), [features[:, i].detach().numpy() for i in steps],
         'time', 'x', legend=[f'{i}-step preds' for i in steps],
         xlim=[5, 1000], figsize=(6, 3))
plt.show()        