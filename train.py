import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from mydataset import MyDataset
from mymodel import MyModel
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    train_dataset = MyDataset("./datasets/train/")
    # shuffle: mess up the order
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    mymodel = MyModel().cuda()
    loss_fn = nn.MultiLabelSoftMarginLoss().cuda()  # 多标签交叉熵损失函数
    # 优化器 Adam 一般要求学习率比较小
    optim = Adam(mymodel.parameters(),
                 lr=0.001  # learn (speed) rate
                 )
    writer = SummaryWriter("logs")
    total_step = 0
    # train for 10 rounds
    for epoch in range(10):
        # train once
        for i, (image, label) in enumerate(train_dataloader):
            image = image.cuda()
            label = label.cuda()
            mymodel.train()
            output = mymodel(image)
            loss = loss_fn(output, label)
            optim.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算
            optim.step()
            total_step += 1
            # display results every 10 runs
            if i % 10 == 0:
                print("epoch {}, step {}, loss {}".format(epoch, i, loss))
                writer.add_scalar("loss", loss, total_step)
    writer.close()
    torch.save(mymodel, "model.pth")
