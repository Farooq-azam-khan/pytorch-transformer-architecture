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

    # Batch Matrix Multiplication
    print('Batch Matrix Multiplication (bij,bjk->bik)')
    batch_size = 3 
    As = torch.randn(batch_size,2,5)
    Bs = torch.randn(batch_size,5,4)
    print(f'{As=}')
    print(f'{Bs=}')
    print(torch.einsum('bij,bjk->bik', As, Bs))


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
