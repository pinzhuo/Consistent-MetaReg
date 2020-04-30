import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert (A.dim() == 3)
    assert (B.dim() == 3)
    assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1, 2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.

    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.gesv(id_matrix, b_mat)

    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape(
        [matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(
        matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

class CM_R2d2Head(nn.Module):

    def __init__(self):
        super(CM_R2d2Head, self).__init__()

    def forward(self, query, support, support_labels, n_way, n_shot):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)

        self.query = query
        self.support = support

        assert (query.dim() == 3)
        assert (support.dim() == 3)
        assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

        id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        # Compute the dual form solution of the ridge regression.
        # W = X^T(X X^T - lambda * I)^(-1) Y
        ridge_sol = computeGramMatrix(support, support) + 1 * id_matrix
        ridge_sol = binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
        self.ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

        # Compute the classification score.
        # score = W X
        logits = torch.bmm(query, self.ridge_sol)

        return logits

    def CMloss_Fnorm(self, query_labes, n_way, n_shot):

        tasks_per_batch = self.query.size(0)
        n_query = self.query.size(1)

        assert (n_query == n_way * n_shot)  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(query_labes.view(tasks_per_batch * n_query), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_query, n_way)

        id_matrix = torch.eye(n_query).expand(tasks_per_batch, n_query, n_query).cuda()

        query_ridge_sol = computeGramMatrix(self.query, self.query) + 1 * id_matrix
        query_ridge_sol = binv(query_ridge_sol)
        query_ridge_sol = torch.bmm(self.query.transpose(1, 2), query_ridge_sol)
        query_ridge_sol = torch.bmm(query_ridge_sol, support_labels_one_hot)
        dis_matrix = query_ridge_sol - self.ridge_sol
        revese_loss = 0
        for i in range(tasks_per_batch):
            revese_loss += (torch.trace(torch.mm(dis_matrix[i,:,:], dis_matrix[i,:,:].t()))).sqrt()

        revese_loss = 1.0 * revese_loss / (n_way* tasks_per_batch)

        return revese_loss

class CM_SVMHead(nn.Module):
    def __init__(self, C_reg=0.1, double_precision=False, maxIter=15):
        super(CM_SVMHead, self).__init__()

        self.C_reg = C_reg
        self.double_precision = double_precision
        self.maxIter = maxIter
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, query, support, support_labels, n_way, n_shot):
        """
        Fits the support set with multi-class SVM and
        returns the classification score on the query set.

        This is the multi-class SVM presented in:
        On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
        (Crammer and Singer, Journal of Machine Learning Research 2001).

        This model is the classification head that we use for the final version.
        Parameters:
          query:  a (tasks_per_batch, n_query, d) Tensor.
          support:  a (tasks_per_batch, n_support, d) Tensor.
          support_labels: a (tasks_per_batch, n_support) Tensor.
          n_way: a scalar. Represents the number of classes in a few-shot classification task.
          n_shot: a scalar. Represents the number of support examples given per class.
          C_reg: a scalar. Represents the cost parameter C in SVM.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        n_query = query.size(1)

        assert (query.dim() == 3)
        assert (support.dim() == 3)
        assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

        # Here we solve the dual problem:
        # Note that the classes are indexed by m & samples are indexed by i.
        # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
        # s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

        # where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        # and C^m_i = C if m  = y_i,
        # C^m_i = 0 if m != y_i.
        # This borrows the notation of liblinear.

        # \alpha is an (n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(support, support)

        id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        # This seems to help avoid PSD error from the QP solver.
        block_kernel_matrix += 1.0 * torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support,
                                                                         n_way * n_support).cuda()

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support),
                                         n_way)  # (tasks_per_batch * n_support, n_support)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)

        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        # print (G.size())
        # This part is for the inequality constraints:
        # \alpha^m_i <= C^m_i \forall m,i
        # where C^m_i = C if m  = y_i,
        # C^m_i = 0 if m != y_i.
        id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * support_labels_one_hot)
        # print (C.size(), h.size())
        # This part is for the equality constraints:
        # \sum_m \alpha^m_i=0 \forall i
        id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, n_support))
        # print (A.size(), b.size())
        if self.double_precision:
            G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
        else:
            G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False, maxIter=self.maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(),
                                                            b.detach())

        # Compute the classification score.
        compatibility = computeGramMatrix(support, query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1)

        self.support_w = torch.bmm(qp_sol.transpose(1,2), support)

        return logits

    def CMloss_Fnorm(self, query, support, support_labels, n_way, n_shot):

        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        n_query = query.size(1)

        assert (query.dim() == 3)
        assert (support.dim() == 3)
        assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

        # Here we solve the dual problem:
        # Note that the classes are indexed by m & samples are indexed by i.
        # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
        # s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

        # where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        # and C^m_i = C if m  = y_i,
        # C^m_i = 0 if m != y_i.
        # This borrows the notation of liblinear.

        # \alpha is an (n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(support, support)

        id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        # This seems to help avoid PSD error from the QP solver.
        block_kernel_matrix += 1.0 * torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support,
                                                                         n_way * n_support).cuda()

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support),
                                         n_way)  # (tasks_per_batch * n_support, n_support)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)

        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        # print (G.size())
        # This part is for the inequality constraints:
        # \alpha^m_i <= C^m_i \forall m,i
        # where C^m_i = C if m  = y_i,
        # C^m_i = 0 if m != y_i.
        id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * support_labels_one_hot)
        # print (C.size(), h.size())
        # This part is for the equality constraints:
        # \sum_m \alpha^m_i=0 \forall i
        id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, n_support))
        # print (A.size(), b.size())
        if self.double_precision:
            G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
        else:
            G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False, maxIter=self.maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(),
                                                                 b.detach())

        # Compute the classification score.
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
        w_query = torch.bmm(qp_sol.transpose(1,2), support)

        dis_matrix = self.support_w - w_query
        revese_loss = 0
        for i in range(tasks_per_batch):
            revese_loss += (torch.trace(torch.mm(dis_matrix[i,:,:], dis_matrix[i,:,:].t()))).sqrt()

        revese_loss = 1.0 * revese_loss / (n_way* tasks_per_batch)


        return revese_loss