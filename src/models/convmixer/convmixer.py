import os
import math
import numpy as np

from mindspore import nn, Tensor
import mindspore
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal
from mindspore.common import initializer

if os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d


class ResidualCell(nn.Cell):
    """
    Cell which implements Residual function:

    $$output = x + f(x)$$

    Args:
        cell (Cell): Cell needed to add residual block.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = ResidualCell(nn.Dense(3,4))
    """

    def __init__(self, cell):
        super(ResidualCell, self).__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x


class ConvMixer(nn.Cell):
    def __init__(self,
        n_classes: int = 1000,
        dim: int = 1536,
        depth: int = 20,
        k_size: int = 9,
        patch_size: int = 7):
        super(ConvMixer, self).__init__()

        patch_embed = nn.Conv2d(3, dim, kernel_size = patch_size, stride= patch_size, pad_mode="valid", has_bias=True)
        self.layers = nn.SequentialCell(
            patch_embed,
            nn.GELU(),
            BatchNorm2d(dim),
            *[nn.SequentialCell(
                ResidualCell(nn.SequentialCell(
                    nn.Conv2d(dim, dim, kernel_size = k_size, pad_mode='same',  group = dim , has_bias=True),
                    nn.GELU(),
                    BatchNorm2d(dim))
                ),
                nn.Conv2d(dim, dim, kernel_size = 1, pad_mode='valid' , has_bias=True),
                nn.GELU(),
                BatchNorm2d(dim)
                ) for i in range(depth)],
            )
        
        self.avgpool = P.ReduceMean(keep_dims = False)
        self.flat = nn.Flatten()
        self.head = nn.Dense(dim, n_classes, has_bias = True)
    
        self.init_weights()
    
    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(sigma=0.02),  
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            
            elif isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Conv1d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                                                             
                if cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))      

            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer.initializer(initializer.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(initializer.initializer(initializer.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def construct(self, x):
        x = self.layers(x)
        x = self.avgpool(x, (2,3))
        x = self.flat(x)
        x = self.head(x)
        return x
    

def convmixer_1536_20(args):
    model = ConvMixer()
    return model



if __name__ == "__main__":
    from mindspore import context
    import numpy as np
    from mindspore import ops

    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }

    graph_mode = 0
    context.set_context(mode=mode[graph_mode], device_target="GPU")

    x = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)  

    model = ConvMixer(n_classes = 1000, dim = 1536, depth = 20, k_size = 9, patch_size=7)
    n_parameters = sum(ops.Size()(p) for p in model.get_parameters() if p.requires_grad)

    print("parameters: ", n_parameters)
    y = model(x)

    print(y.shape)