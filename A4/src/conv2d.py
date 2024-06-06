import torch
import torch.nn.functional as F

def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(x, k, b):
    """ Sliding window solution. """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(torch.multiply(window, k))
    return result + b


def pytorch(x, k, b):
    """ PyTorch solution. """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b   # (1, )
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    """ TODO: implement `im2col`"""
    input_height, input_width = x.shape
    kernel_height, kernel_width = k.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    cols = torch.zeros((kernel_height * kernel_width, output_height * output_width))
    col_index = 0
    for i in range(output_height):
        for j in range(output_width):
            patch = x[i:i+kernel_height, j:j+kernel_width]
            cols[:, col_index] = patch.reshape(-1)
            col_index += 1

    k_flat = k.reshape(-1, 1)
    output = cols.t().matmul(k_flat).reshape(output_height, output_width)
    output += b

    return output


def winograd(x, k, b):
    """ TODO: implement `winograd`"""
    G = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.0, 0.0, 1.0]
    ], dtype=x.dtype)

    B = torch.tensor([
        [1, 0, -1, 0],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [0, 1, 0, -1]
    ], dtype=k.dtype)

    A = torch.tensor([
        [1, 1, 1, 0],
        [0, 1, -1, -1]
    ], dtype=b.dtype)

    input_height, input_width = x.shape
    output_height = input_height - k.shape[0] + 2
    output_width = input_width - k.shape[1] +2
    padded_input = F.pad(x, (0, 1, 0, 1))

    output = torch.zeros(output_height, output_width, dtype=x.dtype)
    for row in range(0, output_height, 2):
        for col in range(0, output_width, 2):
            U = G @ k @ G.T
            V = B @ padded_input[row:row+4, col:col+4] @ B.T
            Y = U * V
            output_block = A @ Y @ A.T
            output[row:row+2, col:col+2] += output_block

    return output[:-1, :-1] + b


def fft(x, k, b):
    """ TODO: implement `fft`"""
    input_height, input_width = x.shape
    kernel_height, kernel_width = k.shape
    
    pad_height = input_height + 2 * (kernel_height - 1)
    pad_width = input_width + 2 * (kernel_width - 1)

    padded_x = torch.zeros(pad_height, pad_width, dtype=x.dtype)
    padded_x[kernel_height-1 : kernel_height-1+input_height, kernel_width-1 : kernel_width-1+input_width] = x

    #flipping kernel for convolution
    flipped_k = torch.flip(k, [0, 1])

    padded_k = torch.zeros(pad_height, pad_width, dtype=k.dtype)
    padded_k[:kernel_height, :kernel_width] = flipped_k

    #FFT of input and kernel
    input_fft = torch.fft.fft2(padded_x)
    kernel_fft = torch.fft.fft2(padded_k)
    
    # multiplication in the freq domain
    fft_mul = input_fft * kernel_fft
    
    #get the spatial domain output
    inv_fft = torch.fft.ifft2(fft_mul).real

    # Crop
    output = inv_fft[2*(kernel_height-1):2*(kernel_height-1)+input_height-kernel_height+1, 
               2*(kernel_width-1):2*(kernel_width-1)+input_width-kernel_width+1]
    
    output += b
    return output