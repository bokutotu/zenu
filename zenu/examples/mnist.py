import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import csv


class MnistCSVDataset(Dataset):
    """
    mnist_train_flattened.txt などの入力画像データと
    mnist_train_labels.txt などのラベルデータを読み込む Dataset 実装例です。
    """
    def __init__(self, data_path: str, label_path: str):
        self.data = []
        # 画像データの読み込み（1行目がヘッダなのでスキップ）
        with open(data_path, "r", newline="") as f_data:
            reader_data = csv.reader(f_data)
            next(reader_data)  # ヘッダをスキップ
            data_lines = list(reader_data)

        # ラベルデータの読み込み（こちらはヘッダの有無に応じて適宜調整）
        with open(label_path, "r", newline="") as f_label:
            reader_label = csv.reader(f_label)
            label_lines = list(reader_label)

        # 行数が同じ前提で、(ピクセル配列, ラベル) をまとめて格納
        for line_data, line_label in zip(data_lines, label_lines):
            # ピクセル値を 0〜255 の u8 相当から float に変換
            # Rust の実装例では (値 / 127.5) - 1.0 で -1〜1 に正規化
            pixels = list(map(float, line_data))  # 文字列 → float
            label = int(line_label[0])           # ラベル（0〜9）
            self.data.append((pixels, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # Tensor に変換 & 正規化
        x = torch.tensor(x, dtype=torch.float32) / 127.5 - 1.0
        # CrossEntropyLoss を使う場合、ラベルは one-hot ではなく整数カテゴリにしておく
        y = torch.tensor(y, dtype=torch.long)
        return x, y


class SimpleModel(nn.Module):
    """
    Rust で定義していた SimpleModel<f32, D> 相当。
    28×28=784 次元→512 次元→10 次元の MLP。
    """
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(784, 512)
        self.linear_2 = nn.Linear(512, 10)

    def forward(self, x):
        # x は [N, 784] 形状想定
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x


def main():
    # Dataset の作成
    train_data_path = "./zenu/examples/mnist_train_flattened.txt"
    train_label_path = "./zenu/examples/mnist_train_labels.txt"
    test_data_path = "./zenu/examples/mnist_test_flattened.txt"
    test_label_path = "./zenu/examples/mnist_test_labels.txt"

    # 1) 訓練＋バリデーション用データセット
    full_dataset = MnistCSVDataset(train_data_path, train_label_path)
    # 2) テストデータセット
    test_dataset = MnistCSVDataset(test_data_path, test_label_path)

    # train : val = 8 : 2 の割合で分割（Rust コードの train_val_split と同様）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader の作成
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデル、損失関数、オプティマイザの準備
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # 学習ループ
    num_epochs = 20
    for epoch in range(num_epochs):
        # -------------------
        # Train
        # -------------------
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 損失の集計
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # -------------------
    # Test
    # -------------------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()

