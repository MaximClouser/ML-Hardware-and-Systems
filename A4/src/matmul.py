import torch
import torch.nn.functional as F

def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    if rank_A:
        U, S, V = torch.svd(A)
        A = torch.matmul(U[:, :rank_A], torch.matmul(torch.diag(S[:rank_A]), V[:, :rank_A].t()))
        
    if rank_B:
        U, S, V = torch.svd(B)
        B = torch.matmul(U[:, :rank_B], torch.matmul(torch.diag(S[:rank_B]), V[:, :rank_B].t()))

    result = torch.matmul(A, B)
    return result


def logmatmul(A, B, **kwargs):
    """ TODO: use log multiplication for matrix-matrix multiplication """
    sign_A, log_A = torch.sign(A), torch.log2(torch.abs(A))
    sign_B, log_B = torch.sign(B), torch.log2(torch.abs(B))

    log_sums = log_A.unsqueeze(2) + log_B.unsqueeze(0)
    signs = sign_A.unsqueeze(2) * sign_B.unsqueeze(0)

    # log product and sum along the 1st dim
    result = torch.sum(signs * (2 ** log_sums), dim=1)
    return result


def nvidia_logmatmul(A, B):
    """
    EXTRA CREDIT: matrix multiplication using logarithmic-based arithmetic as described by NVIDIA.
    """
    M, N = A.shape
    _, K = B.shape
    n = 2  # base factor
    result = torch.zeros((M, K), dtype=A.dtype, device=A.device)
    for i in range(M):
        for j in range(K):
            # partial sums for each possible remainder component
            partial_sums = torch.zeros(n, dtype=A.dtype, device=A.device)
            for k in range(N):
                #multiplication in log domain
                a_mult_n = (A[i, k] * n).int()
                b_mult_n = (B[k, j] * n).int()

                #quotient and remainder
                a_eq = a_mult_n // n
                a_er = a_mult_n % n
                b_eq = b_mult_n // n
                b_er = b_mult_n % n

                #total exponent and remainder
                total_eq = a_eq + b_eq
                total_er = (a_er + b_er) % n

                # Accumulate contributions in the respective partial sum bucket
                partial_sums[total_er] += 2 ** total_eq.float()

            # Combine partial sums
            combined_sums = torch.sum(partial_sums * (2 ** (torch.arange(n, dtype=torch.float, device=A.device) / n)))
            result[i, j] = torch.log2(combined_sums)
    return result
