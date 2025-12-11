import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  
from PIL import Image  

def add_noise_snr(image, snr_db):
    original_device = image.device  
    image_np = image.cpu().numpy()  
    P_signal = np.mean(image_np ** 2)
    if P_signal == 0:
        return image.to(original_device)  
    if snr_db == np.inf:
        return image.to(original_device)
    sigma_sq = P_signal / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(sigma_sq), image_np.shape)
    noisy_np = np.clip(image_np + noise, 0, 1)
    return torch.from_numpy(noisy_np).float().to(original_device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_model(model, loader, device, snr_db=np.inf):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            if snr_db != np.inf:
                noisy_data = torch.stack([add_noise_snr(img, snr_db) for img in data])
            else:
                noisy_data = data
            output = model(noisy_data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    accuracy = correct / total
    return accuracy, all_preds, all_targets

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_PATH = 'mnist_cnn_model.pth'

    required_files = [
        'training_history.png',
        'confusion_matrix_clean.png',
        'accuracy_vs_snr.png',
        'hist_snr_inf.png',
        'hist_snr_30.png',
        'hist_snr_20.png',
        'hist_snr_10.png',
        'hist_snr_0.png',
        'hist_snr_-5.png'
    ]

    all_files_exist = os.path.exists(MODEL_PATH) and all(os.path.exists(f) for f in required_files)

    if all_files_exist:
        print("Все файлы (модель и графики) уже существуют")
        model = CNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    else:
        print("Не все файлы существуют")
        
        model = CNN().to(device)

        if os.path.exists(MODEL_PATH):
            print("Файл модели найден")
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            print("Модель успешно загружена")
        else:
            print("Файл модели не найден")

            transform_train = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()


            train_losses, train_accuracies = [], []
            for epoch in range(10):
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                avg_loss = running_loss / len(train_loader)
                accuracy = correct / total
                train_losses.append(avg_loss)
                train_accuracies.append(accuracy)
                print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Модель обучена и сохранена как '{MODEL_PATH}'.")

            if not os.path.exists('training_history.png'):
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(train_accuracies, label='Train Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig('training_history.png')
                plt.show()
            else:
                print("✔️ График обучения 'training_history.png' уже существует. Пропускаем.")

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        if not os.path.exists('confusion_matrix_clean.png'):
            clean_acc, preds, targets = evaluate_model(model, test_loader, device)
            print(f"Clean Test Accuracy: {clean_acc * 100:.2f}%")

            cm = confusion_matrix(targets, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig('confusion_matrix_clean.png')
            plt.show()

            print(classification_report(targets, preds))
        else:
            print("Матрица ошибок 'confusion_matrix_clean.png' уже существует")

        snr_levels = [np.inf, 30, 20, 10, 0, -5]

        if not os.path.exists('accuracy_vs_snr.png'):
            noisy_accuracies = []
            for snr in snr_levels:
                acc, _, _ = evaluate_model(model, test_loader, device, snr)
                noisy_accuracies.append(acc)
                print(f"SNR={snr} dB, Accuracy: {acc:.4f}")

            plt.figure()
            plt.plot(snr_levels, noisy_accuracies, marker='o')
            plt.xlabel('SNR (dB)')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs SNR')
            plt.grid(True)
            plt.savefig('accuracy_vs_snr.png')
            plt.show()
        else:
            print("График 'accuracy_vs_snr.png' уже существует")

        num_trials = 30
        mc_results = {snr: [] for snr in snr_levels}
        perform_mc = any(not os.path.exists(f'hist_snr_{snr if snr != np.inf else "inf"}.png') for snr in snr_levels)
        
        if perform_mc:
            for snr in snr_levels:
                hist_path = f'hist_snr_{snr if snr != np.inf else "inf"}.png'
                if os.path.exists(hist_path):
                    print(f"Гистограмма для SNR={snr} уже существует")
                    continue
                
                for _ in range(num_trials):
                    acc, _, _ = evaluate_model(model, test_loader, device, snr)
                    mc_results[snr].append(acc)
                mean_acc = np.mean(mc_results[snr])
                std_acc = np.std(mc_results[snr])
                print(f"SNR={snr} dB: Mean Acc={mean_acc:.4f} ± {std_acc:.4f}")

                plt.figure()
                plt.hist(mc_results[snr], bins=10)
                plt.xlabel('Accuracy')
                plt.ylabel('Frequency')
                plt.title(f'Histogram of Accuracy at SNR={snr} dB')
                plt.savefig(hist_path)
                plt.show()
        else:
            print("Все гистограммы Монте-Карло уже существуют")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        img = Image.open('sample_digit.jpg').convert('L')  
        img = transform(img).unsqueeze(0).to(device)  
        output = model(img)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        probability = prob[0][pred].item()
        print(f"Модель предсказывает: Цифра {pred}")
        print(f"Вероятность: {probability:.4f}")
    except FileNotFoundError:
        print("Файл 'sample_digit.jpg' не найден")
    
    print(model)