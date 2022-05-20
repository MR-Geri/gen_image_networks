import torch.nn as nn


def gen_layers__start_finish_step(start: int, finish: int, step: int, activation=nn.Tanh) -> tuple[nn.Sequential, str]:
    layers = [nn.Linear(2, start, bias=True), activation()]
    for i in range(start, finish, -step):
        ney = nn.Linear(i, i - step, bias=False)
        layers += [ney, activation()]
    layers += [nn.Linear(finish, 3, bias=False), nn.Sigmoid()]
    return nn.Sequential(*layers), f'{activation.__name__}__start_{start}__finish_{finish}__step_{step}'


def gen_layers__num(num: int, activation=nn.Tanh) -> tuple[nn.Sequential, str]:
    layers = [nn.Linear(2, num, bias=True), activation()]
    for _ in range(num - 1):
        layers += [nn.Linear(num, num, bias=False), activation()]
    layers += [nn.Linear(num, 3, bias=False), nn.Sigmoid()]
    return nn.Sequential(*layers), f'{activation.__name__}__num_{num}'


def gen_layers__degree_two(degree: int, activation=nn.Tanh) -> tuple[nn.Sequential, str]:
    layers = [nn.Linear(2, 2 ** degree, bias=True), activation()]
    for i in range(degree, 2, -1):
        layers += [nn.Linear(2 ** i, 2 ** (i - 1), bias=False), activation()]
    layers += [nn.Linear(4, 3, bias=False), nn.Sigmoid()]
    return nn.Sequential(*layers), f'{activation.__name__}__degree_{degree}'


def gen_image(activation, quality_sl, quality_ne, sig: bool = False):
    layers = [nn.Linear(2, quality_ne, bias=True), activation()]
    for i in range(quality_sl):
        layers += [nn.Linear(quality_ne, quality_ne, bias=False), activation()]
    layers += [nn.Linear(quality_ne, 3, bias=False), nn.Sigmoid() if sig else activation()]
    return nn.Sequential(*layers), f'{activation.__name__}__{quality_ne}__{quality_sl}'

