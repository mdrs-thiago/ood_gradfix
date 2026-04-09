import torch  
from sklearn.decomposition import PCA 
import numpy as np

class OODMethod:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def compute_ood_scores(self, inputs):
        raise NotImplementedError("Subclasses should implement this method.")
    
class MSP(OODMethod):
    def compute_ood_scores(self, loader):
        ood_scores = []
        with torch.no_grad():
            for inputs, _ in loader:
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                ood_scores.append(1 - max_probs)
        return torch.cat(ood_scores, dim=0)

class EnergyBased(OODMethod):
    def compute_ood_scores(self, loader):
        ood_scores = []
        with torch.no_grad():
            for inputs, _ in loader:
                outputs = self.model(inputs)
                energy_scores = -torch.logsumexp(outputs, dim=1)
                ood_scores.append(energy_scores)
        return torch.cat(ood_scores, dim=0)

class Odin(OODMethod):
    def __init__(self, model, epsilon=0.001):
        super().__init__(model)
        self.epsilon = epsilon

    def compute_ood_scores(self, loader):
        ood_scores = []
        for inputs, labels in loader:
            inputs.requires_grad = True
            outputs = self.model(inputs)
            max_scores, _ = torch.max(outputs, dim=1)
            loss = -torch.mean(max_scores)
            loss.backward()
            perturbed_inputs = inputs + self.epsilon * inputs.grad.sign()
            with torch.no_grad():
                outputs_perturbed = self.model(perturbed_inputs)
                probabilities = torch.softmax(outputs_perturbed, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                ood_scores.append(1 - max_probs)
        return torch.cat(ood_scores, dim=0)

class GradVec(OODMethod):
    def __init__(self, model, agg_method='mean'):
        super().__init__(model)
        self.agg_method = agg_method
    def extract_gradients(self, input):
        executed = []

        def fwd_hook(name):
            def _hook(module, inp, out):
                # Record only modules that have trainable params
                if any(p.requires_grad for p in module.parameters(recurse=False)):
                    executed.append(name)
            return _hook

        handles = []
        for name, m in self.model.named_modules():
            if len(list(m.children())) == 0:
                handles.append(m.register_forward_hook(fwd_hook(name)))

        self.model.zero_grad()
        self.model.train()
        output = self.model(input)
        prob = torch.softmax(output, dim=1)
        uniform_dist = torch.ones_like(prob) / prob.size(1)
        loss = torch.nn.functional.cross_entropy(output, uniform_dist, reduction="none")
        loss.backward(gradient=torch.ones_like(loss))
        for h in handles:
            h.remove()
        if not executed:
            raise RuntimeError("No executed module with trainable parameters was found.")
        last_name = executed[-1]
        last_module = dict(self.model.named_modules())[last_name]
        grads = {}
        for name, p in last_module.named_parameters(recurse=False):
            grads[name] = None if p.grad is None else p.grad.detach().clone()
        return grads['weights'] if 'weights' in grads else grads['weight']
    
    def fit(self, loader):
        gradient_list = []
        labels = []
        for batch in loader:
            # loader must yield (inputs, labels) tuples for fitting
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, batch_labels = batch
            else:
                raise RuntimeError("GradVec.fit requires a dataloader that yields (inputs, labels) tuples.")

            # If inputs are batched, compute per-sample gradients so labels align 1:1
            if torch.is_tensor(inputs) and inputs.dim() > 1 and inputs.size(0) > 1:
                for i in range(inputs.size(0)):
                    inp = inputs[i:i+1]
                    grads = self.extract_gradients(inp)
                    if self.agg_method == 'flatten':
                        grads = grads.view(-1)
                    elif self.agg_method == 'mean':
                        grads = grads.mean(dim=0)
                    elif self.agg_method == 'sum':
                        grads = grads.sum(dim=0)
                    else:
                        raise ValueError(f"Unknown aggregation method: {self.agg_method}")
                    gradient_list.append(grads)
                    labels.append(batch_labels[i])
            else:
                # single-sample input or non-tensor input
                grads = self.extract_gradients(inputs)
                if self.agg_method == 'flatten':
                    grads = grads.view(-1)
                elif self.agg_method == 'mean':
                    grads = grads.mean(dim=0)
                elif self.agg_method == 'sum':
                    grads = grads.sum(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {self.agg_method}")
                gradient_list.append(grads)
                # batch_labels could be a scalar tensor or single-element tensor
                if torch.is_tensor(batch_labels) and batch_labels.numel() > 1:
                    labels.append(batch_labels.view(-1)[0])
                else:
                    labels.append(batch_labels)

        np_gradients = torch.stack(gradient_list).cpu().numpy()
        # Use dimensionality reduction on gradient_list considering 95% explained variance
        try:
            pca = PCA(n_components=0.95)
            grad_lowdim = pca.fit_transform(np_gradients)
            self.pca = pca
        except Exception:
            grad_lowdim = np_gradients
            self.pca = None
        # For each class, fit a Gaussian to the gradients
        self.class_stats = {}
        # convert collected labels to a flat numpy array
        flat_labels = []
        for l in labels:
            if torch.is_tensor(l):
                flat_labels.extend(l.detach().cpu().reshape(-1).tolist())
            else:
                flat_labels.append(int(l))
        labels_array = np.array(flat_labels)
        if labels_array.shape[0] != grad_lowdim.shape[0]:
            raise RuntimeError(f"Mismatch between gradients ({grad_lowdim.shape[0]}) and labels ({labels_array.shape[0]})")
        for cls in np.unique(labels_array):
            cls_grads = grad_lowdim[labels_array == cls]
            mean = np.mean(cls_grads, axis=0)
            cov = np.cov(cls_grads, rowvar=False) + 1e-6 * np.eye(cls_grads.shape[1])  # add small value to diagonal for stability
            self.class_stats[cls] = (mean, cov)
    def compute_ood_scores(self, loader):
        ood_scores = []
        for batch in loader:
            # support (inputs, labels) or inputs-only
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _ = batch
            else:
                inputs = batch

            # batched inputs: compute per-sample scores
            if torch.is_tensor(inputs) and inputs.dim() > 1 and inputs.size(0) > 1:
                for i in range(inputs.size(0)):
                    grads = self.extract_gradients(inputs[i:i+1])
                    if self.agg_method == 'flatten':
                        grads = grads.view(-1).cpu().numpy()
                    elif self.agg_method == 'mean':
                        grads = grads.mean(dim=0).cpu().numpy()
                    elif self.agg_method == 'sum':
                        grads = grads.sum(dim=0).cpu().numpy()
                    else:
                        raise ValueError(f"Unknown aggregation method: {self.agg_method}")
                    if self.pca is not None:
                        grads_lowdim = self.pca.transform(grads.reshape(1, -1))
                    else:
                        grads_lowdim = grads.reshape(1, -1)
                    min_mahalanobis = float('inf')
                    for mean, cov in self.class_stats.values():
                        diff = grads_lowdim[0] - mean
                        inv_cov = np.linalg.inv(cov)
                        mahalanobis_dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
                        if mahalanobis_dist < min_mahalanobis:
                            min_mahalanobis = mahalanobis_dist
                    ood_scores.append(torch.tensor([min_mahalanobis]))
            else:
                grads = self.extract_gradients(inputs)
                if self.agg_method == 'flatten':
                    grads = grads.view(-1).cpu().numpy()
                elif self.agg_method == 'mean':
                    grads = grads.mean(dim=0).cpu().numpy()
                elif self.agg_method == 'sum':
                    grads = grads.sum(dim=0).cpu().numpy()
                else:
                    raise ValueError(f"Unknown aggregation method: {self.agg_method}")
                if self.pca is not None:
                    grads_lowdim = self.pca.transform(grads.reshape(1, -1))
                else:
                    grads_lowdim = grads.reshape(1, -1)
                min_mahalanobis = float('inf')
                for mean, cov in self.class_stats.values():
                    diff = grads_lowdim[0] - mean
                    inv_cov = np.linalg.inv(cov)
                    mahalanobis_dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
                    if mahalanobis_dist < min_mahalanobis:
                        min_mahalanobis = mahalanobis_dist
                ood_scores.append(torch.tensor([min_mahalanobis]))
        return torch.cat(ood_scores, dim=0)

if __name__ == '__main__':
    # Example usage considering a model with three layers
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(20, 10)
            self.fc3 = torch.nn.Linear(10, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    model = SimpleModel()
    gradvec_method = GradVec(model)
    # Testing with two random inputs 

    samples = [(torch.randn(1, 10), torch.tensor(i%5)) for i in range(100)]
    gradvec_method.fit(samples)
    ood_scores = gradvec_method.compute_ood_scores([torch.randn(1, 10) for _ in range(10)])
    print(ood_scores)
