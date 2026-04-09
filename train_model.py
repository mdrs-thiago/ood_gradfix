import torch 
import time 
from ood_methods import MSP, EnergyBased, Odin, GradVec
from ood_evaluate import compute_auroc, compute_aupr, compute_fpr_at_tpr95

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return model, epoch_loss 

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return epoch_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        init_time = time.time()
        model, train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}'
              f' Time: {time.time() - init_time:.2f}s')
        

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

def train_evaluating_ood(model, train_loader, val_loader, train_loader_1b, test_loader_1b, ood_loader, criterion, optimizer, num_epochs, ood_method_name):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        init_time = time.time()
        model, train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        msp_method = MSP(model)
        id_scores = msp_method.compute_ood_scores(test_loader_1b)
        ood_scores = msp_method.compute_ood_scores(ood_loader)
        auroc = compute_auroc(id_scores, ood_scores)
        aupr = compute_aupr(id_scores, ood_scores)
        fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores)
        print(f'OOD Evaluation MSP - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR@95TPR: {fpr95:.4f}')

        energy_method = EnergyBased(model)
        id_scores = energy_method.compute_ood_scores(test_loader_1b)
        ood_scores = energy_method.compute_ood_scores(ood_loader)
        auroc = compute_auroc(id_scores, ood_scores)
        aupr = compute_aupr(id_scores, ood_scores)
        fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores)
        print(f'OOD Evaluation EnergyBased - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR@95TPR: {fpr95:.4f}')

        # odin_method = Odin(model, epsilon=0.001)
        gradvec_method = GradVec(model, agg_method='mean')
        gradvec_method.fit(train_loader)
        id_scores = gradvec_method.compute_ood_scores(test_loader_1b)
        ood_scores = gradvec_method.compute_ood_scores(ood_loader)
        auroc = compute_auroc(id_scores, ood_scores)
        aupr = compute_aupr(id_scores, ood_scores)
        fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores)
        print(f'OOD Evaluation MSP - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR@95TPR: {fpr95:.4f}')

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}'
              f' Time: {time.time() - init_time:.2f}s')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model




