import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import functions_gen_layers as fgl
import matplotlib.image as img

CUDA, LOAD = False, True
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


def fit(model, data, batch_size=100, train=True):
    model.train(train)
    for xb, yb in data.items():
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


def gen_data_lern(data, size):
    to_neuro = lambda x, y: (2 * x / size[0] - 1, 2 * y / size[1] - 1)
    return {
        torch.Tensor(to_neuro(x, y)).to(device=device, non_blocking=True):
            torch.Tensor(data[x][y]).to(device=device, non_blocking=True)
        for x in range(size[0]) for y in range(size[1])
    }


def get_image_from_neuro(neuro, size):
    get_cord = lambda num: (num % size[0], (num // size[0]) % size[1])
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

    torch.save(state, 'state.pth')  # сохраняем файл


def load():
    state = torch.load('state.pth')
    model = NN(layers).to(device=device, non_blocking=True)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    optimizer.load_state_dict(state['optimizer'])
    print(f'epoch={state["epoch"]} lern_time={state["lern_time"]}')  # вспомогательная информация
    return model, optimizer, state['lern_time'], state["epoch"]


if __name__ == '__main__':
    model = NN(layers).to(device=device, non_blocking=True).apply(init_normal)
    loss = nn.MSELoss().to(device=device, non_blocking=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, nesterov=True)
    Y = img.imread('images/image.jpg') / 255
    SIZE = Y.shape
    #
    start_epoch, epochs = 0, 10000  # число эпох
    data = gen_data_lern(Y, Y.shape)
    start = datetime.datetime.now()
    last = datetime.datetime.now()

    if LOAD:
        model, optimizer, l_time, start_epoch = load()
        start -= l_time

    for epoch in range(start_epoch, epochs):
        now = datetime.datetime.now()
        print(f'\tвремени прошло: {now - start} (+{now - last})')
        last = datetime.datetime.now()
        fit(model, data)  # одна эпоха
        if not epoch % 5:
            show(get_image_from_neuro(model, SIZE), is_save=True)
        if not epoch % 100:
            print(f'{epoch=}')
            # show(get_image_from_neuro(model, SIZE))
            save(epoch, now - start)
