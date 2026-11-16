"""Shape-only tape declarations and tensor regions, without execution or
storage.
"""

import numpy as np

from ygaraty.compiler.tape import Tape


class ShapeTape(Tape):
    def declare(self, name, shape, initializer, parameter=False):
        name = self._name(name)
        self.shapes[name] = tuple(shape)
        self.values[name] = dict(shape=list(shape), **initializer)
        if parameter:
            self.parameters.append(name)
        return name

    def input(self, name, data, *, parameter=False):
        # Reverse-mode construction only needs a scalar seed for this model.
        a = np.asarray(data, dtype=np.float64)
        if a.size > 1024:
            raise ValueError("Use declare() for large shape-only inputs")
        return self.declare(
            name, a.shape, dict(kind="constant", data=a.tolist()), parameter
        )


def regions(shape, tile):
    for i in range(0, shape[0], tile):
        for j in range(0, shape[1], tile):
            yield [i, min(i + tile, shape[0]), j, min(j + tile, shape[1])]
