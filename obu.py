import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import functions_gen_layers as fgl
import matplotlib.image as img


CUDA = False
device = torch.device('cuda')
layers = fgl.gen_image(nn.Sigmoid, 4, 100, True)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class NN(nn.Module):
    def __init__(self, layers: tuple[nn.Sequential, str]):
        super(NN, self).__init__()
        self.layers, self.metadata = layers
        if CUDA:
            self.layers = self.layers.to(device=device)

    def forward(self, x):
        return self.layers(x)


def fit(model, data, batch_size=100, train=True):
    # важно для Dropout, BatchNorm
    model.train(train)
    # ошибка, точность, батчей
    for xb, yb in data.items():
        xb, yb = torch.Tensor(xb), torch.Tensor(yb) 
        # прямое распространение
        y = model(xb)
        L = loss(y, yb)             # вычисляем ошибку

        if train:                   # в режиме обучения
            optimizer.zero_grad()   # обнуляем градиенты
            L.backward()            # вычисляем градиенты
            optimizer.step()        # подправляем параметры


def map_point(point, size):
    return torch.Tensor(list( map(lambda x: 2*x/size-1, point) ))


def show(image):
    plt.imshow(image)
    plt.show()


def gen_data_lern(data, size):
    to_neuro = lambda x, y: (2 * x / size[0] - 1, 2 * y / size[1] - 1)
    return { 
        to_neuro(x, y): data[x][y]
        for x in range(size[0]) for y in range(size[1])
    }


def get_image_from_neuro(neuro, size):
    get_cord = lambda num: (num % size[0], (num // size[0]) % size[1])
    to_neuro = lambda x, y: (2 * x / size[0] - 1, 2 * y / size[1] - 1)
    data = [
        [
            neuro(torch.Tensor(to_neuro(x, y))).detach().numpy() 
            for y in range(size[1])
        ] 
        for x in range(size[0])
    ]
    return data

    
def save(epoch: int):
    state = {'info':      f"эпоха: {epoch}",            # описание
             'date':      datetime.datetime.now(),   # дата и время
             'model' :    model.state_dict(),        # параметры модели
             'optimizer': optimizer.state_dict()}    # состояние оптимизатора
     
    torch.save(state, 'state.pt')                    # сохраняем файл


def load():
    state = torch.load('state.pt')
    model = NN(layers).load_state_dict(state['model'])
    optimizer = torch.optim.SGD(model.parameters(),lr=1).load_state_dict(state['optimizer'])     
    print(state['info'], state['date'])              # вспомогательная информация
    return model, optimizer


if __name__ == '__main__':
    if CUDA:
        model = NN(layers).to(device=device).apply(init_normal)
    else:
        model = NN(layers).apply(init_normal)
    print(model.metadata)

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, nesterov=True)
    Y = img.imread('images/image.jpg') / 255
    SIZE = Y.shape
#
    epochs = 1000                  # число эпох
    data = gen_data_lern(Y, Y.shape)
    for epoch in range(epochs):   # эпоха - проход по всем примерам
        fit(model, data)          # одна эпоха
         
        if epoch % 100 == 0:
            print(epoch)
            show(get_image_from_neuro(model, SIZE))
            save(epoch)

#    print(model(torch.Tensor(X[32][32])))

