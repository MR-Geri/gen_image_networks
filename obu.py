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
functions = (
    (
        nn.Sigmoid, lambda x, y, size: (2 * (x / size[0] - 0.5), 2 * (y / size[1] - 0.5)),
        lambda color: color, lambda color: color
    ),
    (
        nn.Tanh, lambda x, y, size: (2 * (x / size[0] - 0.5), 2 * (y / size[1] - 0.5)),  # хз
        lambda color: color * 2 - 1, lambda color: (color + 1) / 2
    ),
    ()
)
func, to_neuro, to_color, from_color = functions[0]
layers = fgl.gen_image(func, 4, 100, False)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class NN(nn.Module):
    def __init__(self, layers: tuple[nn.Sequential, str]):
        super(NN, self).__init__()
        self.layers, self.metadata = layers
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
                self.input.append(to_neuro(x, y, self.size))
                self.output.append(to_color(image[x][y]))
        self.input = torch.tensor(np.array(self.input), device=device, dtype=torch.float)
        self.output = torch.tensor(np.array(self.output), device=device, dtype=torch.float)

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
    s = datetime.datetime.now()
    plt.imshow(image)
    if is_save:
        plt.savefig(f'images_gen/{datetime.datetime.now():%Y-%m-%d--%H-%M-%S}.jpg')
    plt.show()
    print(f'render = {datetime.datetime.now() - s}')


def get_image_from_neuro(neuro, size):
    data = torch.tensor([
        [to_neuro(x, y, size) for y in range(size[1])] for x in range(size[0])
    ], device=device, dtype=torch.float)
    out = neuro(data).detach().cpu().numpy() if CUDA else neuro(data).detach().numpy()
    out = from_color(out)
    return out


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
    torch.cuda.empty_cache()
    model = NN(layers).to(device=device, non_blocking=True).apply(init_normal)
    loss = nn.MSELoss(reduction='sum').to(device=device, non_blocking=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.7, nesterov=True)
    Y = img.imread('images/image.jpg') / 255
    SIZE = Y.shape
    start = datetime.datetime.now()
    print('size =', SIZE)
    #
    data1 = MyDataset(Y)
    data_load = DataLoader(data1, batch_size=64, shuffle=True)
    # show(np.array(from_color(data1.output.detach().cpu().numpy())).reshape((64, 64, 3)))
    epoch = 0
    print('data done', datetime.datetime.now() - start)
    start = datetime.datetime.now()
    if LOAD:
        model, optimizer, l_time, epoch = load('')
        start -= l_time

    while True:
        epoch += 1
        fit(model, data_load)  # одна эпоха
        now = datetime.datetime.now()
        if not epoch % 100:
            print(f'{epoch=} времени прошло: {now - start}')
        if not epoch % 100 or epoch == 1:
            show(get_image_from_neuro(model, SIZE), is_save=False)
        # if not epoch % 500:
        #     save(epoch, now - start)
