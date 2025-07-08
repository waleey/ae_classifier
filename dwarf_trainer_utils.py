import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.optim.lr_scheduler import CyclicLR

class TrainerUtils:
    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _load_data(self):
        train_data, train_labels, _ = self.trainer.dpc.get_train_set()
        val_data, val_labels, val_tics = self.trainer.dpc.get_validation_set()

        train_data = train_data.to(self.trainer.device)
        train_labels = train_labels.to(self.trainer.device)
        val_data = val_data.to(self.trainer.device)
        val_labels = val_labels.to(self.trainer.device)

        if self.trainer.metadata_dim > 0:
            train_meta = self.trainer.dpc.get_train_meta().to(self.trainer.device)
            val_meta = self.trainer.dpc.get_val_meta().to(self.trainer.device)

            self.trainer.train_loader = DataLoader(
                TensorDataset(train_data, train_labels, train_meta),
                batch_size=self.trainer.batch_size,
                shuffle=False
            )
            self.trainer.val_loader = DataLoader(
                TensorDataset(val_data, val_labels, val_meta, torch.tensor(val_tics)),
                batch_size=self.trainer.batch_size,
                shuffle=False
            )
        else:
            self.trainer.train_loader = DataLoader(
                TensorDataset(train_data, train_labels),
                batch_size=self.trainer.batch_size,
                shuffle=False
            )

            self.trainer.val_loader = DataLoader(
                TensorDataset(val_data, val_labels, torch.tensor(val_tics)),
                batch_size=self.trainer.batch_size,
                shuffle=False
            )

    def _init_scheduler(self):
        if self.trainer.scheduler_type == 'cyclic':
            return CyclicLR(
                self.trainer.optimizer,
                base_lr=1e-5,
                max_lr=0.01,
                step_size_up=3000,
                mode='triangular'
            )
        else:
            return None

    def _init_wb(self, model, a=0.1):
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=a)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def compute_accuracy(self, logits, labels, tics):
        if self.trainer.binary_class:
            probs = torch.sigmoid(logits)
            predicted_classes = (probs > 0.5).long()
        else:
            predicted_classes = torch.argmax(logits, dim=1)

        correct = (predicted_classes == labels.long()).sum().item()

        false_positives = []
        false_negatives = []

        if tics is not None:
            for i in range(len(predicted_classes)):
                pred = predicted_classes[i].item()
                true = labels[i].item()
                tic = tics[i]
                if isinstance(tic, torch.Tensor):
                  tic = tic.item()

                if pred != true:
                    if pred == 1 and true == 0:
                        false_positives.append(tic)
                    elif pred == 0 and true == 1:
                        false_negatives.append(tic)

        return correct, predicted_classes, false_positives, false_negatives

    def get_classification_loss(self):
        binary = self.trainer.binary_class
        weighted = self.trainer.class_weighted_loss
        device = self.trainer.device

        if not weighted:
            return nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

        # Compute class weights from training labels
        all_labels = []
        if self.trainer.metadata_dim > 0:
            for _, labels, meta in self.trainer.train_loader:
                all_labels.extend(labels.cpu().numpy())
                del meta
        else:
            for _, labels in self.trainer.train_loader:
                all_labels.extend(labels.cpu().numpy())

        label_counts = Counter(all_labels)
        num_classes = max(label_counts.keys()) + 1  # safer than len(label_counts)
        total_samples = sum(label_counts.values())

        weights = [total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

        if binary:
            pos_weight = torch.tensor([weights_tensor[1] / weights_tensor[0]], dtype=torch.float).to(device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.CrossEntropyLoss(weight=weights_tensor)


    def _validate(self):
        self.trainer.model.eval()
        val_recon_loss = 0
        correct_preds = 0
        total_samples = 0

        all_preds = []
        all_labels = []
        all_tics = []
        logits = []

        all_recons = []
        all_originals = []

        with torch.no_grad():
            #for batch_data, batch_labels, batch_tics in self.trainer.val_loader:
            for batch in self.trainer.val_loader:
                if self.trainer.metadata_dim > 0:
                    batch_data, batch_labels, batch_meta, batch_tics = batch
                else:
                    batch_data, batch_labels, batch_tics = batch
                    batch_meta = None
                batch_data = batch_data.to(self.trainer.device)
                batch_labels = batch_labels.to(self.trainer.device)
                if batch_meta is not None:
                    batch_meta = batch_meta.to(self.trainer.device)
                    self.trainer.model.set_val_metadata(batch_meta)

                reconstruction, class_logits = self.trainer.model(batch_data)
                recon_loss = self.trainer.recon_loss_fn(reconstruction, batch_data) * batch_data.size(0)
                val_recon_loss += recon_loss.item()

                logits.append(class_logits)
                all_labels.append(batch_labels)
                all_tics.append(batch_tics)
                total_samples += batch_labels.size(0)

                if self.trainer.save_val_recons:
                    all_originals.append(batch_data.cpu())
                    all_recons.append(reconstruction.cpu())

        if self.trainer.save_val_recons and all_recons:
            os.makedirs(self.trainer.output_dir, exist_ok=True)
            original = torch.cat(all_originals, dim=0).squeeze(1).numpy()
            recon = torch.cat(all_recons, dim=0).squeeze(1).numpy()

            df_original = pd.DataFrame(original)
            df_original['type'] = 'original'

            df_recon = pd.DataFrame(recon)
            df_recon['type'] = 'reconstruction'

            df_combined = pd.concat([df_original, df_recon], ignore_index=True)
            csv_path = os.path.join(self.trainer.output_dir, "val_originals_and_reconstructions.csv")
            df_combined.to_csv(csv_path, index=False)

        logits = torch.cat(logits, dim=0)
        device = logits.device
        all_labels = torch.cat(all_labels, dim=0).to(device)
        all_tics = torch.cat(all_tics, dim = 0)


        if self.trainer.binary_class:
            all_labels = all_labels.float()

        val_class_loss = self.trainer.class_loss_fn(logits, all_labels)
        correct_preds, all_preds, self.trainer.false_positives, self.trainer.false_negatives = self.compute_accuracy(logits, all_labels, all_tics)

        self.trainer.all_predicted_labels = all_preds.cpu().numpy()
        self.trainer.all_true_labels = all_labels.cpu().numpy()

        # === Compute precision, recall, F1 ===
        from sklearn.metrics import precision_score, recall_score, f1_score

        if self.trainer.binary_class:
            average_type = 'binary'
        else:
            average_type = 'weighted'

        precision = precision_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average=average_type, zero_division=0)
        recall = recall_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average=average_type, zero_division=0)
        f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average=average_type, zero_division=0)

        return {
            'val_recon_loss': val_recon_loss / len(self.trainer.val_loader),
            'val_class_loss': val_class_loss.item(),
            'val_accuracy': correct_preds / total_samples,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }


    def run_epoch(self, epoch, train_autoencoder=True, train_classifier=True):
        self.trainer.model.train()
        train_recon_loss = 0
        train_class_loss = 0

        #for batch_data, batch_labels in self.trainer.train_loader:
        for batch in self.trainer.train_loader:
            if self.trainer.metadata_dim > 0:
                batch_data, batch_labels, batch_meta = batch
            else:
                batch_data, batch_labels = batch
                batch_meta = None
            batch_data = batch_data.to(self.trainer.device)
            batch_labels = batch_labels.to(self.trainer.device)
            if(batch_meta is not None):
                batch_meta = batch_meta.to(self.trainer.device)
                self.trainer.model.set_train_metadata(batch_meta)

            self.trainer.optimizer.zero_grad()
            reconstruction, class_logits = self.trainer.model(batch_data)

            recon_loss = self.trainer.recon_loss_fn(reconstruction, batch_data) * batch_data.size(0)

            #if self.trainer.class_weighted_loss:
                #self.trainer.class_loss_fn.weight = self.trainer.class_loss_fn.weight.to(class_logits.device)

            if self.trainer.binary_class:
                batch_labels = batch_labels.float()
            class_loss = self.trainer.class_loss_fn(class_logits, batch_labels)

            total_loss = 0
            if train_autoencoder:
                total_loss += recon_loss
            if train_classifier:
                total_loss += class_loss

            total_loss.backward()
            self.trainer.optimizer.step()

            if self.trainer.scheduler:
                self.trainer.scheduler.step()

            train_recon_loss += recon_loss.item()
            train_class_loss += class_loss.item()

        train_recon_loss /= len(self.trainer.train_loader)
        train_class_loss /= len(self.trainer.train_loader)
        self.trainer.train_recon_losses.append(train_recon_loss)
        self.trainer.train_class_losses.append(train_class_loss)

        # Skip validation and checkpointing if we're only training the autoencoder
        if not train_classifier:
            print(f"Epoch [{epoch + 1}/{self.trainer.num_epochs}], "
                f"Train Recon: {train_recon_loss:.4f}")
            return False

        # Perform validation only when classifier is being trained
        val_metrics = self._validate()
        self.trainer.val_recon_losses.append(val_metrics['val_recon_loss'])
        self.trainer.val_class_losses.append(val_metrics['val_class_loss'])
        self.trainer.val_accuracies.append(val_metrics['val_accuracy'])
        self.trainer.val_precisions.append(val_metrics['val_precision'])
        self.trainer.val_recalls.append(val_metrics['val_recall'])
        self.trainer.val_f1_scores.append(val_metrics['val_f1'])

        self.trainer.save_model(epoch, val_metrics['val_recall'])
        self.trainer.last_epoch = epoch

        if self.trainer.training_mode == 'joint':
            val_total_loss = val_metrics['val_recon_loss'] + val_metrics['val_class_loss']
        else:
            val_total_loss = val_metrics['val_class_loss']
        self.trainer.early_stopping(val_total_loss)
        if self.trainer.early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            return True
        self.trainer.best_val_recall = self.trainer.early_stopping.save_best_checkpoint(
            self.trainer.model,
            val_metrics['val_recall'],
            best_recall=self.trainer.best_val_recall,
            path=os.path.join(self.trainer.output_dir, 'best_model.pth')
        )

        print(f"Epoch [{epoch + 1}/{self.trainer.num_epochs}], "
            f"Train Recon: {train_recon_loss:.4f}, Train Class: {train_class_loss:.4f}, "
            f"Val Recon: {val_metrics['val_recon_loss']:.4f}, Val Class: {val_metrics['val_class_loss']:.4f}, "
            f"Val Recall: {val_metrics['val_recall']:.4f},", 
            f"Val accuracy: {val_metrics['val_accuracy']:.4f}, ")

        return False

