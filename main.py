import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import functions_gen_layers as fgl
import functions_activate as fa


CUDA = False
device = torch.device('cuda')
# torch.cuda.set_per_process_memory_fraction(0.7, 0)


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


def gen_new_image(size_x, size_y, layers: tuple[nn.Sequential, str]):
    if CUDA:
        net = NN(layers).to(device=device).apply(init_normal)
    else:
        net = NN(layers).apply(init_normal)
    print(net.metadata)
    try:
        colors = run_net(net, size_x, size_y)
        plot_colors(colors)
        plt.imsave(f"images/{datetime.datetime.now().strftime('%d.%m.%y_%H.%M.%S')}__{net.metadata}.png", colors)
    except:
        print("^^^ NO ^^^")


def run_net(net, size_x, size_y):
    x, y = np.arange(0, size_x, 1), np.arange(0, size_y, 1)
    colors = np.zeros((size_x, size_y, 2))
    for i in x:
        for j in y:
            colors[i][j] = np.array([float(i) / size_y - 0.5, float(j) / size_x - 0.5])
    colors = colors.reshape(size_x * size_y, 2)
    if CUDA:
        img = net(torch.FloatTensor(colors).to(device)).detach().cpu().numpy()
        torch.cuda.empty_cache()
    else:
        img = net(torch.tensor(colors).type(torch.FloatTensor)).detach().numpy()
    return img.reshape(size_x, size_y, 3)


def plot_colors(colors, fig_size=16) -> None:
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)
    plt.show()


def main() -> None:
    gen_new_image(2048, 2048, fgl.gen_layers__start_finish_step(
        start=128, finish=16, step=16, activation=nn.Tanh
    ))
    # for func in (nn.SiLU, fa.Sinh, fa.Cosh, nn.Tanh):
    #     gen_new_image(2048, 2048, fgl.gen_layers__start_finish_step(
    #         start=128, finish=64, step=16, activation=func
    #     ))
    # for degree in range(2, 8):
    #     for func in (nn.SiLU, fa.Sinh, fa.Cosh, nn.Tanh):
    #         gen_new_image(2048, 2048, fgl.gen_layers__degree_two(
    #             degree=degree, activation=func
    #         ))
    # for num in range(2, 17):
    #     for func in (nn.SiLU, fa.Sinh, fa.Cosh, nn.Tanh):
    #         gen_new_image(2048, 2048, fgl.gen_layers__num(
    #             num=num, activation=func
    #         ))


if __name__ == '__main__':
    main()
