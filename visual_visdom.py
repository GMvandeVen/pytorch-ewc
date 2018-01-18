import numpy as np
from torch.cuda import FloatTensor as CUDATensor
from visdom import Visdom

_WINDOW_CASH = {}


def _vis(env='main'):
    return Visdom(env=env)


def visualize_scalars(scalars, names, title, iteration, env='main'):
    assert len(scalars) == len(names)
    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in
               scalars]
    multi = len(scalars) > 1
    num = len(scalars)

    options = dict(
        fillarea=False,
        legend=names,
        width=400,
        height=400,
        xlabel='Iterations',
        ylabel=title,
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    )

    X = (
        np.column_stack(np.array([iteration] * num)) if multi else
        np.array([iteration] * num)
    )
    Y = np.column_stack(scalars) if multi else scalars[0]

    if title in _WINDOW_CASH:
        _vis(env).updateTrace(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options)
    else:
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)
