import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
import subprocess
import time

# 모델과 데이터 설정
def prepare_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # IMDb 영화 리뷰 데이터셋 로드
    dataset = load_dataset("imdb")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    test_dataset = tokenized_datasets["test"].shuffle(seed=42)

    return train_dataset, test_dataset

# 학습 루프
def train_model(train_dataset, device):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = torch.nn.DataParallel(model)  # 여러 GPU 사용
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 2

    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()

            # 입력 데이터 준비
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward 및 Backward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 배치 처리 시간 측정
            elapsed_time = time.time() - start_time
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Time per batch: {elapsed_time:.2f}s")

    print("Training Done!")

def main():
    # 모니터링 스크립트를 별도 프로세스로 실행
    monitor_process = subprocess.Popen(["python3", "monitor.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Started resource monitoring...")

    try:
        # 데이터 준비
        train_dataset, _ = prepare_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 학습
        train_model(train_dataset, device)
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # 모니터링 종료
        monitor_process.terminate()
        monitor_process.wait()
        print("Stopped resource monitoring.")

if __name__ == "__main__":
    main()
