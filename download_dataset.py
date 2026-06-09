
import torchvision

# download=True 会自动下载 RGB 版本的 EuroSAT 到当前目录的 data 文件夹下
dataset = torchvision.datasets.EuroSAT(root="./data", download=True)
print("下载完成！数据集大小：", len(dataset))