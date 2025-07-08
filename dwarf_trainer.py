import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import ae_classifier_combined
from early_stopping import EarlyStopping
from dwarf_trainer_utils import TrainerUtils
class DwarfTrainer:
    def __init__(self, data_processor, model_id=None, input_shape=1024,
                 ae_model='nets_1024', classifier_model='mlp',
                 batch_size=64, num_epochs=1500, lr=1e-3, classifier_lr = 1e-4, scheduler_type='cyclic',
                 init_weights=True, leaky_relu_a=0.1, seed=42,
                 min_delta=1e-2, patience=20, save_val_recons=False,
                 class_weighted_loss=False,
                 training_mode='joint', ae_epochs=500):
        
        self.dpc = data_processor
        self.metadata_dim = self.dpc.get_metadata_dim()
        self.model_id = model_id
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.ae_epochs = ae_epochs
        self.lr = lr
        self.classifier_lr = classifier_lr
        self.scheduler_type = scheduler_type
        self.leaky_relu_a = leaky_relu_a
        self.seed = seed
        self.min_delta = min_delta
        self.patience = patience
        self.save_val_recons = save_val_recons
        self.class_weighted_loss = class_weighted_loss
        self.binary_class = True if classifier_model == 'binary_classifier' else False
        self.training_mode = training_mode
        self.old_epochs = 0
        self.last_epoch = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(seed)

        # Output directory
        self.output_dir = f"../saved_model/model_{self.model_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize training utilities (this will also load data)
        self.utils = TrainerUtils()
        self.utils.set_trainer(self)
        self.utils._load_data()

        # Initialize model
        self.model = ae_classifier_combined.autoencoder_classifier(
            ae_model=ae_model,
            classifier_model=classifier_model,
            output_shape=len(self.dpc.get_label_mapping()),
            metadata_dim=self.metadata_dim
        ).to(self.device)

        if init_weights:
            self.utils._init_wb(self.model, self.leaky_relu_a)

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = self.utils._init_scheduler()

        # Loss functions
        self.recon_loss_fn = nn.MSELoss()
        self.class_loss_fn = self.utils.get_classification_loss()

        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # Tracking metrics
        self.train_recon_losses = []
        self.train_class_losses = []
        self.val_recon_losses = []
        self.val_class_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        self.false_positives = []
        self.false_negatives = []

    def train_and_validate(self):
        self.best_val_recall = 0.0

        if self.training_mode == 'sequential':
            print(f"Sequential training mode: first training autoencoder for {self.ae_epochs} epochs.")

            # Phase 1: Autoencoder pretraining
            for epoch in range(self.ae_epochs):
                stop = self.utils.run_epoch(epoch, train_autoencoder=True, train_classifier=False)
                if stop:
                    break

            # Freeze encoder
            print("Freezing encoder and training classifier only.")
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            if self.classifier_lr != 0.0:
                self.lr = self.classifier_lr
            # Re-initialize optimizer and scheduler for classifier-only phase
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            self.scheduler = self.utils._init_scheduler()

            # Phase 2: Classifier training
            
            for epoch in range(self.ae_epochs, self.num_epochs):
                stop = self.utils.run_epoch(epoch, train_autoencoder=False, train_classifier=True)
                if stop:
                    break

        else:
            print("Joint training mode: training autoencoder and classifier together.")
            for epoch in range(self.num_epochs):
                stop = self.utils.run_epoch(epoch, train_autoencoder=True, train_classifier=True)
                if stop:
                    break

        print("Training complete.")

    def set_lr(self, lr):
        self.lr = lr
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def set_classifier_lr(self, lr):
        self.classifier_lr = lr
        if self.classifier_lr != 0.0:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.classifier_lr)

    def set_attr_(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid attribute of DwarfTrainer.")
            
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_recon_losses.append(checkpoint['train_recon_loss'])
        self.train_class_losses.append(checkpoint['train_class_loss'])
        self.val_recon_losses.append(checkpoint['val_recon_loss'])
        self.val_class_losses.append(checkpoint['val_class_loss'])
        self.val_accuracies.append(checkpoint['val_accuracy'])
        self.old_epochs = checkpoint['epoch']
        self.model.to(self.device)
        print(f"Model loaded from {model_path}. Last epoch: {self.old_epochs}")

    def load_best_model(self, path = None):
        if path is None:
            path = os.path.join(self.output_dir, 'best_model.pth')
        
        checkpoint = torch.load(path, map_location=self.device)

        # If checkpoint is a full dict (common), extract the model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If you saved the raw model state dict directly
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        print(f"Loaded model from {path}")

    def plot_losses(self):
        #Saving the loss functions
        path = os.path.join(self.output_dir, "loss_logs.txt")
        with open(path, "w") as f:
          f.write("TrainRecon\tValRecon\tTrainClass\tValClass\n")
          for t_recon, v_recon, t_class, v_class in zip(
          self.train_recon_losses,
          self.val_recon_losses,
          self.train_class_losses,
          self.val_class_losses
          ):
            f.write(f"{t_recon:.6f}\t{v_recon:.6f}\t{t_class:.6f}\t{v_class:.6f}\n")

        # Reconstruction loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_recon_losses, label='Train Recon Loss')
        plt.plot(self.val_recon_losses, label='Val Recon Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss')
        plt.legend()

        # Classification loss
        plt.subplot(1, 3, 2)
        plt.plot(self.train_class_losses, label='Train Class Loss')
        plt.plot(self.val_class_losses, label='Val Class Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classification Loss')
        plt.legend()

        #total loss
        plt.subplot(1, 3, 3)
        total_train_loss = np.array(self.train_recon_losses) + np.array(self.train_class_losses)
        total_val_loss = np.array(self.val_recon_losses) + np.array(self.val_class_losses)
        plt.plot(total_train_loss, label='Total Train Loss')
        plt.plot(total_val_loss, label='Total Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(6, 5))
        print(f"Plotting accuracies for {len(self.val_accuracies)} epochs.")
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.show()

    def plot_precision(self):
        plt.figure(figsize=(6, 5))
        print(f"Plotting precisions for {len(self.val_precisions)} epochs.")
        plt.plot(self.val_precisions, label='Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Validation Precision')
        plt.legend()
        plt.show()

    def plot_recall(self):
        plt.figure(figsize=(6, 5))
        print(f"Plotting recalls for {len(self.val_recalls)} epochs.")
        plt.plot(self.val_recalls, label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Validation Recall')
        plt.legend()
        plt.show()

    def plot_f1_score(self):
        plt.figure(figsize=(6, 5))
        print(f"Plotting F1 scores for {len(self.val_f1s)} epochs.")
        plt.plot(self.val_f1s, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self):
      # Reverse the label mapping: int -> string
      index_to_classname = {idx: name for idx, name in self.dpc.get_label_mapping().items()}

      # Create class names list in correct order
      class_names = [index_to_classname[i] for i in range(len(index_to_classname))]

      cm = confusion_matrix(self.all_true_labels, self.all_predicted_labels, normalize='true')
      disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

      plt.figure(figsize=(8, 6))
      disp.plot(cmap="Blues", xticks_rotation=45)
      plt.title("Validation Confusion Matrix")
      plt.tight_layout()
      path = os.path.join(self.output_dir, f"confusion_matrix.png")
      plt.savefig(path, dpi = 300)
      plt.show()

    def save_fp_and_fn(self):
      fp_path = os.path.join(self.output_dir, "false_positives.txt")
      fn_path = os.path.join(self.output_dir, "false_negatives.txt")

      with open(fp_path, "w") as f:
          for tic in self.false_positives:
              f.write(f"{tic}\n")

      with open(fn_path, "w") as f:
          for tic in self.false_negatives:
              f.write(f"{tic}\n")

    def save_model(self, epoch, val_accuracy):
        model_path = os.path.join(self.output_dir, f"model_latest.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_recon_loss': self.train_recon_losses[-1],
            'train_class_loss': self.train_class_losses[-1],
            'val_recon_loss': self.val_recon_losses[-1],
            'val_class_loss': self.val_class_losses[-1],
            'val_accuracy': val_accuracy,
            'epoch': epoch
        }, model_path)
    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def get_accuracy(self):
        return self.val_accuracies

    def get_predicted_labels(self):
        return self.all_predicted_labels
    def get_true_labels(self):
        return self.all_true_labels
    def write_input_description(self):
        input_path = os.path.join(self.output_dir, "input_dec.txt")
        with open(input_path, "w") as f:
            f.write("DwarfTrainer Input Configuration\n")
            f.write("=" * 40 + "\n")
            f.write(f"Model ID: {self.model_id}\n")
            f.write(f"Input Shape: {self.input_shape}\n")
            f.write(f"AE Model: {self.model.__class__.__name__}\n")
            f.write(f"Classifier Model: {self.model.classifier.__class__.__name__}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"AE Epochs: {self.ae_epochs}\n")
            f.write(f"Total Epochs: {self.num_epochs}\n")
            f.write(f"Learning Rate: {self.lr}\n")
            f.write(f"Scheduler Type: {self.scheduler_type}\n")

            # Safe check for whether encoder has trainable weights
            encoder_trainable = any(p.requires_grad for p in self.model.encoder.parameters())
            f.write(f"Encoder Weights Trainable: {encoder_trainable}\n")

            f.write(f"Leaky ReLU Alpha: {self.leaky_relu_a}\n")
            f.write(f"Random Seed: {self.seed}\n")
            f.write(f"EarlyStopping Patience: {self.patience}\n")
            f.write(f"EarlyStopping Min Delta: {self.min_delta}\n")
            f.write(f"Save Validation Reconstructions: {self.save_val_recons}\n")
            f.write(f"Class Weighted Loss: {self.class_weighted_loss}\n")
            f.write(f"Binary Classification: {self.binary_class}\n")
            f.write(f"Training Mode: {self.training_mode}\n")
            f.write(f"Device: {self.device}\n")
            f.write("=" * 40 + "\n")

