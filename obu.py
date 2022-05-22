import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import functions_gen_layers as fgl
import matplotlib.image as img

CUDA, LOAD = True, False
device = torch.device('cuda' if CUDA else 'cpu')
layers = fgl.gen_image(nn.Sigmoid, 4, 100, True)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class NN(nn.Module):
    def __init__(self, layers: tuple[nn.Sequential, str]):
        super(NN, self).__init__()
        self.layers, self.metadata = layers
        self.flatten = nn.Flatten()
        if CUDA:
            self.layers = self.layers.to(device=device, non_blocking=True)

    def forward(self, x):
        return self.layers(x)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = image.shape
        self.input = []
        self.output = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                self.input.append(torch.Tensor(self.to_neuro(x, y)).to(device=device, non_blocking=True))
                self.output.append(torch.Tensor(image[x][y]).to(device=device, non_blocking=True))

    def to_neuro(self, x, y):
        return 2 * x / self.size[0] - 1, 2 * y / self.size[1] - 1

    def __len__(self):
        return self.size[0] * self.size[1]

    def __getitem__(self, num):
        return self.input[num], self.output[num]


def fit(model, data, train=True):
    model.train(train)
    for xb, yb in data:
        # прямое распространение
        y = model(xb)
        L = loss(y, yb)  # вычисляем ошибку

        if train:  # в режиме обучения
            optimizer.zero_grad()  # обнуляем градиенты
            L.backward()  # вычисляем градиенты
            optimizer.step()  # подправляем параметры


def show(image, is_save: bool = False):
    plt.imshow(image)
    if is_save:
        plt.savefig(f'images_gen/{datetime.datetime.now():%Y-%m-%d--%H-%M-%S}.jpg')
    plt.show()


def get_image_from_neuro(neuro, size):
    to_neuro = lambda x, y: (2 * x / size[0] - 1, 2 * y / size[1] - 1)
    data = [
        [
            neuro(torch.Tensor(to_neuro(x, y)).to(device=device, non_blocking=True)).detach().cpu().numpy()
            if CUDA else
            neuro(torch.Tensor(to_neuro(x, y))).detach().numpy()
            for y in range(size[1])
        ]
        for x in range(size[0])
    ]
    return data


def save(epoch: int, lern_time):
    state = {'epoch': epoch,  # описание
             'lern_time': lern_time,
             'model': model.state_dict(),  # параметры модели
             'optimizer': optimizer.state_dict()}  # состояние оптимизатора

    torch.save(state, f'state_{str(lern_time).replace(":", "_")}.pth')  # сохраняем файл
    print(f'SAVE')


def load(path: str):
    state = torch.load(path)
    model = NN(layers).to(device=device, non_blocking=True)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    optimizer.load_state_dict(state['optimizer'])
    print(f'epoch={state["epoch"]} lern_time={state["lern_time"]}')  # вспомогательная информация
    return model, optimizer, state['lern_time'], state["epoch"]


if __name__ == '__main__':
    model = NN(layers).to(device=device, non_blocking=True).apply(init_normal)
    loss = nn.MSELoss(reduction='sum').to(device=device, non_blocking=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, nesterov=True)
    Y = img.imread('images/image.jpg') / 255
    SIZE = Y.shape
    #
    data1 = MyDataset(Y)
    data = DataLoader(data1, batch_size=64, shuffle=True)

    epoch, start_epoch, epochs = 0, 0, 10 ** 6  # число эпох
    start = datetime.datetime.now()

    if LOAD:
        model, optimizer, l_time, start_epoch = load('')
        start -= l_time

    while True:
        epoch += 1
        now = datetime.datetime.now()
        fit(model, data)  # одна эпоха
        if not epoch % 10:
            print(f'{epoch=} времени прошло: {now - start}')
            show(get_image_from_neuro(model, SIZE), is_save=True)
        # if not epoch % 500:
        #     save(epoch, now - start)
