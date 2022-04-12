import torch 


def main():
    matrix = torch.rand(4,4)
    print(matrix)
    print('Trace of Above matrix (ii).')
    # https://en.wikipedia.org/wiki/Trace_(linear_algebra)
    print(torch.einsum('ii', matrix))
    print('Diagonal (ii->i)')
    print(torch.einsum('ii->i', matrix))
    
    x = torch.randn(5)
    y = torch.randn(4)
    print(f'{x=}')
    print(f'{y=}')
    # https://en.wikipedia.org/wiki/Outer_product
    print('Outer Product (i,j->ij)')
    print(torch.einsum('i,j->ij', x, y))
    y = torch.randn(4)
    
    # Matrix Multiplication
    print('Matrix Multiplication (ij,ik->ik)')
    print(torch.einsum('ij,ik->ik', matrix, matrix))


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
