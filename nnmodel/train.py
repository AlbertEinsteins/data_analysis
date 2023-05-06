import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
import torch.optim as optim

from network import NNClassifier
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor()]),

    "test": transforms.Compose([transforms.ToTensor()])}

data_root = os.path.abspath(os.getcwd())  # get data root path


train_dataset = datasets.FashionMNIST(root=data_root, train=True, transform=data_transform['train'], download=True)
test_dataset = datasets.FashionMNIST(root=data_root, train=False, transform=data_transform['test'])

train_num = len(train_dataset)
test_num = len(test_dataset)

clothes_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in clothes_list.items())

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)



batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
# load pretrain weights
net = NNClassifier(n_classes=len(clothes_list))
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

best_acc = 0.0
save_path = './pretrained.pth'

if os.path.exists(save_path):
    net.load_state_dict(torch.load(save_path))

for epoch in range(1):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data

        images = images.repeat(1, 3, 1, 1)

        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in test_dataloader:
            val_images, val_labels = val_data
            val_images = val_images.repeat(1, 3, 1, 1)

            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / test_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
          (epoch + 1, 0 / len(train_loader), val_accurate))
