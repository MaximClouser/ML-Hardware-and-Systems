import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                # TODO: fill-in (start)
                weight_flops = module.weight.numel() * 2
                bias_flops = module.bias.numel() if module.bias is not None else 0
                flops[name] = (weight_flops + bias_flops) * batch_size
                # TODO: fill-in (end)

            if isinstance(module, nn.Conv2d):
                # TODO: fill-in (start)
                batch_size, _, H_out, W_out = output.shape
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_height, kernel_width = module.kernel_size
                flops_per_instance = 2 * in_channels * kernel_height * kernel_width * out_channels * H_out * W_out
                
                if module.bias is not None:
                    bias_flops = out_channels * H_out * W_out
                    flops_per_instance += bias_flops
                    
                flops[name] = flops_per_instance * batch_size
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                # TODO: fill-in (end)
                batch_size, num_features, length = input[0].shape
                flops[name] = 4 * num_features * length * batch_size
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm2d):
                # TODO: fill-in (end)
                batch_size, num_features, height, width = input[0].shape
                flops[name] = 4 * num_features * height * width * batch_size
                # TODO: fill-in (end)
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    # TODO: fill-in (start)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # TODO: fill-in (end)


def calculate_model_size(model):
  """
  Returns the total model size (flash size) based on the paramers
  """
  return sum(p.numel() * p.element_size() for p in model.parameters())


def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    model.to(device)

    input_tensor = torch.rand(input_shape).to(device)
    total_memory = 0

    # Memory used by the input
    total_memory += input_tensor.element_size() * input_tensor.nelement()

    with torch.no_grad():
        output_tensor = model(input_tensor)

        # Memory used by the output
        total_memory += output_tensor.element_size() * output_tensor.nelement()

    return total_memory


    # # TODO: fill-in (start)

    # # Find memory used by inputs and outputs using hooks
    # model.to(device)
    # input_tensor = torch.rand(input_shape).to(device)
    # total_memory = 0

    # def hook(module, input, output):
    #     nonlocal total_memory
    #     input_memory = sum([inp.element_size() * inp.nelement() for inp in input])
    #     output_memory = output.element_size() * output.nelement()
    #     total_memory += (input_memory + output_memory)

    # handles = []
    # for layer in model.children():
    #     handle = layer.register_forward_hook(hook)
    #     handles.append(handle)

    # with torch.no_grad():
    #     model(input_tensor)

    # for handle in handles:
    #     handle.remove()

    # # Add memory used by paramaters
    # # param_memory = sum([param.nelement() * param.element_size() for param in model.parameters()])
    # # total_memory += param_memory

    # return total_memory
    # # TODO: fill-in (end)
