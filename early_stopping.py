import torch 

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    """
    def save_best_checkpoint(self, model, val_accuracy, best_accuracy, path='best_model.pt'):
        if val_accuracy > best_accuracy:
            #print(f"New best model saved at epoch {self.last_epoch+1} with accuracy {val_accuracy:.4f}")
            torch.save(model.state_dict(), path)
            return val_accuracy  # update best_accuracy
        return best_accuracy
    """   
    def save_best_checkpoint(self, model, val_recall, best_recall, path='best_model.pt'):
        if val_recall > best_recall:
            #print(f"New best model saved at epoch {self.last_epoch+1} with recall {val_recall:.4f}")
            torch.save(model.state_dict(), path)
            return val_recall
        return best_recall
    
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.last_epoch = 0
        self.best_accuracy = 0.0
        self.best_recall = 0.0
