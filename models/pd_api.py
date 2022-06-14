import paddle


def transpose(tensor, s, d):
    shape = list(range(tensor.ndim))
    shape[s], shape[d] = shape[d], shape[s]
    y = paddle.transpose(tensor, perm=shape)
    return y


def split(tensor, ssos, dim):
    dim_size = tensor.shape[dim]
    if isinstance(ssos, list):
        num_or_sections = ssos
    else:
        num_or_sections = []
        if dim_size % ssos == 0:
            for _ in range(dim_size // ssos):
                num_or_sections.append(ssos)
        else:
            for _ in range(dim_size // ssos):
                num_or_sections.append(ssos)
            num_or_sections.append(dim_size % ssos)

    return paddle.split(tensor, num_or_sections=num_or_sections, axis=dim)
