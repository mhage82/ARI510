import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

SEED = 42
MODEL_NAME = "bert-base-uncased"


def embed_texts(texts, tokenizer, encoder, batch_size=16):
    # Same BERT CLS embedding approach as Lab 3, with 128 tokens per text.
    embeddings = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[start:start + batch_size], padding=True,
                           truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            embeddings.append(encoder(**inputs).last_hidden_state[:, 0, :].cpu())
    return torch.cat(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("premature-problem-solving"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    train = pd.read_csv(args.data_dir / "train.csv", dtype={"id": str})
    validation = pd.read_csv(args.data_dir / "validation.csv", dtype={"id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"id": str})
    if list(sample.columns) != ["id", "label"] or sample.id.duplicated().any() or set(sample.id) != set(test.id):
        raise ValueError("Sample submission IDs or columns do not match the test set")

    # The weights must already be downloaded; no private text is sent to an API.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).eval()

    def embed_pairs(frame):
        context = embed_texts(frame.context.tolist(), tokenizer, encoder)
        response = embed_texts(frame.response.tolist(), tokenizer, encoder)
        return torch.cat((context, response), dim=1)

    print("Creating BERT embeddings for train, validation, and test...")
    X_train = embed_pairs(train)
    X_val = embed_pairs(validation)
    X_test = embed_pairs(test)
    y_train = torch.tensor(train.label.to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(validation.label.to_numpy(), dtype=torch.float32).unsqueeze(1)

    # Same small feedforward classifier, loss, and optimizer as Lab 3.
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(X_train.shape[1], 128), nn.ReLU(), nn.Linear(128, 1))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(15):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        val_predictions = (torch.sigmoid(model(X_val)).squeeze(1) >= 0.5).int().numpy()
        test_predictions = (torch.sigmoid(model(X_test)).squeeze(1) >= 0.5).int().numpy()
    macro_f1 = f1_score(validation.label, val_predictions, average="macro", zero_division=0)
    accuracy = accuracy_score(validation.label, val_predictions)
    print(f"Best epoch: {best_epoch}; validation macro-F1={macro_f1:.4f}; accuracy={accuracy:.4f}")

    predictions = pd.DataFrame({"id": test.id, "label": test_predictions})
    submission = sample[["id"]].merge(predictions, on="id", validate="one_to_one", how="left")
    if len(submission) != len(test) or submission.label.isna().any() or not set(submission.label).issubset({0, 1}):
        raise ValueError("Invalid submission")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_dir / "run2_bert_mlp.csv", index=False)


if __name__ == "__main__":
    main()
