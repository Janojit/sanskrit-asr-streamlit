#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Sanskrit ASR training script using precomputed Wav2Vec2 features.
"""

import os
import sys
import math
import json
import random
from glob import glob
from datetime import datetime
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from jiwer import wer
from tokenizers import Tokenizer

# -------------------------------*************************--------------------------------
# -------------------------------configuration section -------------------------------- ---
# -------------------------------*************************--------------------------------
train_enable = True if 10 else False  # keeps the original intent (always True)

target_start_token_idx = 7
target_end_token_idx = 8

batch_size = 64
max_target_len = 100
tokenizer_file = "./tokenizer/tokenizer-sanskrit.json"

wav2vec_feature_dir = "./dataset/wav2vec_features"
transcript_file = "./dataset/text.txt"
feature_hidden_size = 1024  # Wav2Vec2-large hidden dimension

# Transformer hyperparameters
num_hid = 128
num_head = 8
num_feed_forward = 512
source_maxlen = 100  # retained for positional embeddings if needed later
target_maxlen = max_target_len
num_layers_enc = 6
num_layers_dec = 6
drop_out_enc = 0.1
drop_out_dec = 0.1
drop_out_cross = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
native_scheduler = True  # True = Vaswani warm-up schedule
lr_log_every = 200  # set to None or 0 to disable intra-epoch LR logging
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ------------------------------------------------------------------------------------------
# Tokenizer vectorizer
# ------------------------------------------------------------------------------------------
class VectorizeChar:
    def __init__(self, max_len):
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.max_len = max_len

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        ids = self.tokenizer.encode(text).ids
        if len(ids) > self.max_len:
            print(f"\nmax target string length greater than set threshold ({self.max_len}): {ids}", file=sys.stderr,)
            sys.exit(-1)
        pad_len = self.max_len - len(ids)
        return ids + [0] * pad_len

    def get_vocabulary_size(self):
        return self.tokenizer.get_vocab_size()

    def idx_to_token(self):
        return self.tokenizer.id_to_token


vectorizer = VectorizeChar(max_target_len)

with open("./results.txt", "a", encoding="utf-8") as file:
    file.write("\n------------------------------------------\n")
    file.write(f"\nResults computed on {datetime.now()} \n")


# ------------------------------------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------------------------------------
class SanskritASRDataset(Dataset):
    def __init__(self, feature_root, script_file, max_text_len):
        self.feature_root = feature_root
        feature_paths = glob(os.path.join(self.feature_root, "**", "*.pt"), recursive=True)

        if not feature_paths:
            raise RuntimeError(
                f"\nNo precomputed feature files found under '{self.feature_root}'. \n"
                f"\nRun the precomputation script before training.\n"
            )

        random.Random(1853).shuffle(feature_paths)
        self.feature_paths = feature_paths

        self.id_to_text = {}
        with open(script_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    idx, text = parts[0], parts[1]
                    self.id_to_text[idx] = text

        self.samples = self._pairup_features_and_script(maxlen=250)
        self.vectorizer = vectorizer
        self.max_text_len = max_text_len

    def _pairup_features_and_script(self, maxlen=250):
        data = []
        for feat_path in self.feature_paths:
            rel_path = os.path.relpath(feat_path, self.feature_root)
            base = os.path.splitext(os.path.basename(feat_path))[0]
            idx = base
            if idx in self.id_to_text and len(self.id_to_text[idx]) < maxlen:
                data.append({"feature": feat_path, "text": self.id_to_text[idx]})
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        payload = torch.load(item["feature"])
        features = payload["features"].to(torch.float32)  # (seq_len, hidden_size)
        attention_mask = payload["attention_mask"].to(torch.bool)  # (seq_len,)
        text_vec = self.vectorizer(item["text"])

        return {
            "source": features,
            "source_mask": attention_mask,
            "target": torch.tensor(text_vec, dtype=torch.long),
        }


def collate_fn(batch):
    sources = [sample["source"] for sample in batch]
    source_masks = [sample["source_mask"] for sample in batch]
    targets = torch.stack([sample["target"] for sample in batch])

    padded_sources = pad_sequence(sources, batch_first=True)  # (batch, max_seq_len, hidden_size)
    padded_masks = pad_sequence(source_masks, batch_first=True, padding_value=False)  # True where valid

    return {
        "source": padded_sources,
        "source_mask": padded_masks,
        "target": targets,
    }


# ------------------------------------------------------------------------------------------
# Transformer building blocks
# ------------------------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab, max_len, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(num_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        return x


class PrecomputedFeatureEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout, cross_dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout_self = nn.Dropout(dropout)

        self.enc_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=cross_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_enc = nn.Dropout(cross_dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    @staticmethod
    def generate_causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        seq_len = tgt.size(1)
        causal_mask = self.generate_causal_mask(seq_len, tgt.device)
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=causal_mask, need_weights=False)
        tgt = self.norm1(tgt + self.dropout_self(attn_output))

        cross_attn_output, _ = self.enc_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout_enc(cross_attn_output))

        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout_ffn(ffn_output))
        return tgt


class TransformerASR(nn.Module):
    def __init__(
        self,
        num_hid,
        num_head,
        num_feed_forward,
        source_maxlen,
        target_maxlen,
        num_layers_enc,
        num_layers_dec,
        num_classes,
        feature_dim,
        drop_out_enc=0.1,
        drop_out_dec=0.1,
        drop_out_cross=0.2,
    ):
        super().__init__()
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.feature_emb = PrecomputedFeatureEmbedding(feature_dim, num_hid)
        self.token_emb = TokenEmbedding(num_classes + 1, target_maxlen, num_hid)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(num_hid, num_head, num_feed_forward, drop_out_enc) for _ in range(num_layers_enc)]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    num_hid,
                    num_head,
                    num_feed_forward,
                    drop_out_dec,
                    drop_out_cross,
                )
                for _ in range(num_layers_dec)
            ]
        )
        self.classifier = nn.Linear(num_hid, num_classes)

    def encode(self, source, source_mask=None):
        x = self.feature_emb(source)
        padding_mask = None
        if source_mask is not None:
            padding_mask = ~source_mask.bool()
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return x, padding_mask

    def decode(self, memory, target, memory_key_padding_mask=None):
        y = self.token_emb(target)
        for layer in self.decoder_layers:
            y = layer(y, memory, memory_key_padding_mask=memory_key_padding_mask)
        return y

    def forward(self, source, target, source_mask=None):
        memory, memory_key_padding_mask = self.encode(source, source_mask=source_mask)
        dec_out = self.decode(memory, target, memory_key_padding_mask=memory_key_padding_mask)
        return self.classifier(dec_out)

    @torch.no_grad()
    def generate(self, source, start_token_idx, end_token_idx=None, source_mask=None):
        batch_size = source.size(0)
        memory, memory_key_padding_mask = self.encode(source, source_mask=source_mask)
        dec_input = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)

        for _ in range(self.target_maxlen - 1):
            dec_output = self.decode(memory, dec_input, memory_key_padding_mask=memory_key_padding_mask)
            logits = self.classifier(dec_output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_token], dim=1)

            if end_token_idx is not None:
                finished |= next_token.squeeze(1).eq(end_token_idx)
                if finished.all():
                    break
        return dec_input


# ------------------------------------------------------------------------------------------
# Callbacks/Monitoring utilities
# ------------------------------------------------------------------------------------------
class DisplayOutputs:
    def __init__(self, dataloader, idx_to_token, target_start, target_end):
        self.dataloader = dataloader
        self.idx_to_token = idx_to_token
        self.target_start = target_start
        self.target_end = target_end
        self.samples = []
        for batch in self.dataloader:
            src = batch["source"][:5]
            src_mask = batch["source_mask"][:5]
            tgt = batch["target"][:5]
            self.samples.append((src, src_mask, tgt))
            break  # only need first batch

    def idxs_to_text(self, idxs):
        tokens = []
        for idx in idxs:
            idx = int(idx)
            if idx == 0:
                continue
            tok = self.idx_to_token(idx)
            tokens.append(tok)
            if idx == self.target_end:
                break
        return "".join(tokens).replace("-", "")

    def log_epoch(self, epoch, loss_value, model):
        with open("./results.txt", "a", encoding="utf-8") as file:
            file.write(f"\nEpoch {epoch + 1}: Training Loss = {loss_value:.4f}")

        if (epoch + 1) % 5 == 0 and self.samples:
            src, src_mask, tgt = self.samples[0]
            src = src.to(device)
            mask = src_mask.to(device)
            preds = model.generate(src, self.target_start, self.target_end, source_mask=mask).cpu()
            with open("./sample_predictions.txt", "a", encoding="utf-8") as out_file:
                out_file.write(f"\n========== Epoch {epoch + 1} ==========\n")
                for i in range(src.size(0)):
                    target_text = self.idxs_to_text(tgt[i].numpy())
                    pred_text = self.idxs_to_text(preds[i].numpy())
                    out_file.write(f"\nSample {i + 1}: \n Target     : {target_text} \n Predicted  : {pred_text}\n")
            print(f"\n[Epoch {epoch + 1}] Logged 5-sample predictions and loss.\n")


class EarlyStoppingAtMinLoss:
    def __init__(self, patience=0):
        self.patience = patience
        self.best = math.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def step(self, metric, epoch):
        if metric < self.best:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                print("Early stopping triggered.")


# ------------------------------------------------------------------------------------------
# Learning-rate schedulers
# ------------------------------------------------------------------------------------------
def get_vaswani_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class CustomSchedule:
    def __init__(self, init_lr, lr_after_warmup, final_lr, warmup_epochs, decay_epochs, steps_per_epoch):
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def lr_lambda(self, step):
        epoch = step // max(self.steps_per_epoch, 1)
        warmup_progress = (self.lr_after_warmup - self.init_lr) / max(self.warmup_epochs - 1, 1)
        warmup_lr = self.init_lr + warmup_progress * epoch
        decay_progress = (self.lr_after_warmup - self.final_lr) / max(self.decay_epochs, 1)
        decay_lr = max(self.final_lr, self.lr_after_warmup - (epoch - self.warmup_epochs) * decay_progress)
        return min(warmup_lr, decay_lr) / self.lr_after_warmup


def get_custom_scheduler(optimizer, schedule: CustomSchedule):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule.lr_lambda)


# ------------------------------------------------------------------------------------------
# Training / evaluation helpers
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def compute_WER(dataloader, model, idx_to_token):
    model.eval()
    predictions = []
    targets = []
    for batch in dataloader:
        source = batch["source"].to(device)
        source_mask = batch["source_mask"].to(device)
        target = batch["target"].cpu().numpy()
        preds = model.generate(source, target_start_token_idx, target_end_token_idx, source_mask=source_mask).cpu().numpy()

        for i in range(preds.shape[0]):
            target_text = "".join([idx_to_token(int(idx)) for idx in target[i] if idx != 0])
            prediction = ""
            for idx in preds[i]:
                idx = int(idx)
                if idx == 0:
                    continue
                prediction += idx_to_token(idx)
                if idx == target_end_token_idx:
                    break
            predictions.append(prediction)
            targets.append(target_text)
    return wer(targets, predictions)


def run_epoch(model, dataloader, optimizer, loss_fn, scheduler=None, start_step=0, log_lr_every=None):
    model.train()
    running_loss = 0.0
    count = 0
    global_step = start_step

    optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        source = batch["source"].to(device)
        source_mask = batch["source_mask"].to(device)
        target = batch["target"].to(device)

        dec_in = target[:, :-1]
        dec_target = target[:, 1:]

        logits = model(source, dec_in, source_mask=source_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        global_step += 1

        if scheduler is not None and log_lr_every and (global_step % log_lr_every == 0):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Step {global_step}] lr = {current_lr:.3e}")

        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        count += 1

    return running_loss / max(count, 1), global_step


@torch.no_grad()
def evaluate(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    count = 0
    for batch in dataloader:
        source = batch["source"].to(device)
        source_mask = batch["source_mask"].to(device)
        target = batch["target"].to(device)

        dec_in = target[:, :-1]
        dec_target = target[:, 1:]

        logits = model(source, dec_in, source_mask=source_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))

        running_loss += loss.item()
        count += 1
    return running_loss / max(count, 1)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model weights successfully saved to '{path}'")


# ------------------------------------------------------------------------------------------
# Main training/evaluation script
# ------------------------------------------------------------------------------------------
def main():
    if train_enable:
        feature_train_dir = os.path.join(wav2vec_feature_dir, "train")
        dataset = SanskritASRDataset(feature_train_dir, transcript_file, max_target_len)
        total_files = len(dataset)
        split_index = int(0.9 * total_files)

        train_subset = torch.utils.data.Subset(dataset, list(range(0, split_index)))
        val_subset = torch.utils.data.Subset(dataset, list(range(split_index, total_files)))

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )

        print(f"Total precomputed feature-text pairs: {total_files}")
        print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

        num_classes = vectorizer.get_vocabulary_size()
        loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

        model = TransformerASR(
            num_hid=num_hid,
            num_head=num_head,
            num_feed_forward=num_feed_forward,
            source_maxlen=source_maxlen,
            target_maxlen=target_maxlen,
            num_layers_enc=num_layers_enc,
            num_layers_dec=num_layers_dec,
            num_classes=num_classes,
            feature_dim=feature_hidden_size,
            drop_out_enc=drop_out_enc,
            drop_out_dec=drop_out_dec,
            drop_out_cross=drop_out_cross,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

        if native_scheduler:
            scheduler = get_vaswani_scheduler(optimizer, d_model=num_hid, warmup_steps=4000)
        else:
            steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
            custom_sched = CustomSchedule(
                init_lr=1e-5,
                lr_after_warmup=1e-3,
                final_lr=1e-5,
                warmup_epochs=90,
                decay_epochs=5,
                steps_per_epoch=steps_per_epoch,
            )
            scheduler = get_custom_scheduler(optimizer, custom_sched)

        display_cb = DisplayOutputs(train_loader, vectorizer.idx_to_token(), target_start_token_idx, target_end_token_idx)
        early_stopping = EarlyStoppingAtMinLoss(patience=15)

        with open("./results.txt", "a", encoding="utf-8") as file:
            file.write("\n==========================================================\n")
            file.write(f"\n Model parameters: num_hid {num_hid}, num_head {num_head}, num_feed_forward {num_feed_forward},\n")
            file.write(f"\n  source_maxlen {source_maxlen}, target_maxlen {target_maxlen}, num_layers_enc {num_layers_enc},\n")
            file.write(f"\n  num_layers_dec {num_layers_dec}, num_classes {num_classes}, drop_out_enc {drop_out_enc}, drop_out_dec {drop_out_dec}\n")
            file.write("\n==========================================================\n")

        max_epochs = 100
        history = {"loss": []}
        global_train_step = 0

        for epoch in range(max_epochs):
            train_loss, global_train_step = run_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                scheduler=scheduler,
                start_step=global_train_step,
                log_lr_every=lr_log_every,
            )
            val_loss = evaluate(model, val_loader, loss_fn) if val_loader else float("nan")
            history["loss"].append(train_loss)
            display_cb.log_epoch(epoch, train_loss, model)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.3e}")

            early_stopping.step(val_loss if not math.isnan(val_loss) else train_loss, epoch)
            if early_stopping.should_stop:
                break

        with open("./results.txt", "a", encoding="utf-8") as file:
            file.write(f"\nTraining loss history: {history['loss']}\n")
            file.write(f"\nThe minimum loss is {min(history['loss']):.3f}\n")
            with redirect_stdout(file):
                print("\n------------------------------------------\n")
                print("\n------------------------------------------\n")

        val_loss = evaluate(model, val_loader, loss_fn) if val_loader else float("nan")
        print(f"Validation loss: {val_loss:.4f}")

        os.makedirs("saved_models", exist_ok=True)
        model_path = "saved_models/sanskrit_transformer_asr.pt"
        save_model(model, model_path)

        print("Evaluating model on validation dataset for WER computation...")
        try:
            val_wer = compute_WER(val_loader, model, vectorizer.idx_to_token())
            print(f"\nValidation Loss: {val_loss:.4f}, WER: {val_wer:.4f}")

            with open("./results.txt", "a", encoding="utf-8") as file:
                file.write("\n================ FINAL EVALUATION ================\n")
                file.write(f"\nValidation loss after training: {val_loss:.4f}\n")
                file.write(f"\nFinal Word Error Rate (WER): {val_wer:.4f}\n")
                file.write(f"\nModel saved at: {model_path}\n")
                file.write("\n==================================================\n")
        except Exception as e:
            print(f"\n❌ Error during final evaluation: {e}")
            with open("./results.txt", "a", encoding="utf-8") as file:
                file.write(f"\nError during final evaluation: {e}\n")


if __name__ == "__main__":
    main()