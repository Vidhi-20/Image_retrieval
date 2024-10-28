# src/dbn.py
import torch
import torch.nn as nn
import torch.optim as optim

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))

    def sample_h(self, v):
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        p_v_given_h = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def contrastive_divergence(self, v, k=1):
        v_k = v
        for _ in range(k):
            _, h = self.sample_h(v_k)
            _, v_k = self.sample_v(h)

        positive_grad = torch.matmul(v.t(), self.sample_h(v)[1])
        negative_grad = torch.matmul(v_k.t(), self.sample_h(v_k)[1])

        self.W.data += 0.01 * (positive_grad - negative_grad) / v.size(0)
        self.v_bias.data += 0.01 * torch.mean(v - v_k, dim=0)
        self.h_bias.data += 0.01 * torch.mean(self.sample_h(v)[1] - self.sample_h(v_k)[1], dim=0)

        return torch.mean(torch.sum((v - v_k) ** 2, dim=1))

class DeepBeliefNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(DeepBeliefNetwork, self).__init__()
        self.rbm_layers = []
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i + 1])
            self.rbm_layers.append(rbm)
            setattr(self, f'rbm_layer_{i}', rbm)

    def forward(self, x):
        for rbm in self.rbm_layers:
            x = rbm(x)
        return x

    def train_rbm(self, data_loader, epochs=10, k=1):
        for idx, rbm in enumerate(self.rbm_layers):
            print(f"Training RBM Layer {idx + 1}")
            optimizer = optim.SGD(rbm.parameters(), lr=0.01)
            for epoch in range(epochs):
                epoch_loss = 0
                for batch, _ in data_loader:
                    batch = batch.view(batch.size(0), -1)  # Flatten the image
                    loss = rbm.contrastive_divergence(batch, k)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")
