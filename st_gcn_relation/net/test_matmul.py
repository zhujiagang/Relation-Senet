import torch
from torch.autograd import Variable

xx_1 = torch.FloatTensor([
                        [[1, 2], [4, 5], [7, 8]],
                        [[1, 2], [4, 5], [7, 8]],
                        [[1, 2], [4, 5], [7, 8]],
                        [[1, 2], [4, 5], [7, 8]]
                          ])


xx_2 = torch.FloatTensor([
                        [[1, 2, 3], [4, 5, 6]],
                        [[1, 2, 3], [4, 5, 6]],
                        [[1, 2, 3], [4, 5, 6]],
                        [[1, 2, 3], [4, 5, 6]]
                          ])

xxx = xx_1 * xx_2

yyy = torch.matmul(xx_1, xx_2)
yyyy = torch.bmm(xx_1, xx_2)
for i in range(4):
    for j in range(3):
        for k in range(3):
            if (yyy[i][j][k] != yyyy[i][j][k]):
                print (xx_1)