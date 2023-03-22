import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.face_dataset import face_dataset
from src.vgg16nn import Vgg16

device = torch.device("cuda:0")
print(device)

train_data = face_dataset(img_dir="../nnimg/test/shuzi/",
                          csv_dir="../csv/Num_test.csv",
                          transform=torchvision.transforms.ToTensor())

test_data = face_dataset(img_dir="../nnimg/test/shuzi/",
                         csv_dir="../csv/Num_test.csv",
                         transform=torchvision.transforms.ToTensor())

train_dataset_size = len(train_data)
test_dataset_size = len(test_data)
print("训练数据集的长度为:{}".format(train_dataset_size))
print("测试数据集的长度为:{}".format(test_dataset_size))

train_data_loader = DataLoader(train_data, batch_size=32, drop_last=True)
test_data_loader = DataLoader(test_data, batch_size=32, drop_last=True)

vgg16 = Vgg16()
vgg16 = vgg16.to(device)

# 损失函数
loss_f = nn.CrossEntropyLoss()
loss_f = loss_f.to(device)
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10
writer = SummaryWriter("logs")
for i in range(epoch):
    print("-------第{}轮训练开始--------".format(i + 1))
    # 训练步骤开始
    for data in train_data_loader:
        train_imgs, train_targets = data
        # 用GPU算
        train_imgs = train_imgs.to(device)
        train_targets = train_targets.to(device)

        outputs = vgg16(train_imgs)
        # 计算损失值
        loss = loss_f(outputs, train_targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{} ,loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracyrate = 0
    with torch.no_grad():
        for data in test_data_loader:
            test_imgs, testtargets = data
            # 用GPU算
            test_imgs = test_imgs.to(device)
            testtargets = testtargets.to(device)

            outputs = vgg16(test_imgs)
            loss = loss_f(outputs, testtargets)
            total_test_loss = total_test_loss + loss
            # 准确率
            accuracyrate = (outputs.argmax(1) == testtargets).sum()
            total_accuracyrate = total_accuracyrate + accuracyrate

    print("整体测试集的Loss:{}".format(total_test_loss))
    print("整体测试集的正确率:{}".format(total_accuracyrate / test_dataset_size))

    writer.add_scalar("test_loss", total_test_loss, global_step=total_test_step)
    writer.add_scalar("test_accuracyrate", total_accuracyrate / test_dataset_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(vgg16, "../trained_mod/vgg16_{}.pth".format(i))
    print("模型已保存")

writer.close()
