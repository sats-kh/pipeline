import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from my_net import *
import subprocess

# 데이터셋 설정
train_set = datasets.MNIST('./mnist_data', download=True, train=True,
               transform=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)

train_epoch = 2

def main():
    # 모니터링 스크립트를 서브프로세스로 실행
    monitor_process = subprocess.Popen(["python3", "monitor.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Started resource monitoring...")
    
    model = MyNet()
    print("Using ", torch.cuda.device_count(), "GPUs for data parallel training")
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
    model = torch.nn.DataParallel(model)
    model.to(device)

    # 학습 루프
    try:
        for epoch in range(train_epoch):
            print(f"Epoch {epoch}")
            for idx, (data, target) in enumerate(train_loader):
                # 데이터 전송
                data, target = data.to(device), target.to(device)

                # Forward 및 Backward
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 로그 출력
                print(f"Batch {idx}, Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # 학습 종료 후 모니터링 스크립트 종료
        monitor_process.terminate()
        monitor_process.wait()
        print("Stopped resource monitoring.")
        print("Training Done!")

if __name__ == '__main__':
    main()
