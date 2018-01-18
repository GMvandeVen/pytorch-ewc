from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=400,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        # Layers.
        self.layers = nn.ModuleList([
            # input
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
            # output
            nn.Linear(self.hidden_size, self.output_size)
        ])

    @property
    def name(self):
        return (
            'MLP'
            '-in{input_size}-out{output_size}'
            '-h{hidden_size}x{hidden_layer_num}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
        ).format(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset):
        '''Estimates diagonal of Fisher Information matrix based on [dataset].

        [dataset]:  list of data-points [x, y] (should be random sample from all previous tasks)'''

        # prepare dictionary to store estimated Fisher Information matrix.
        est_fisher_info = {}
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            est_fisher_info[n] = p.data.clone().zero_()

        # set model to evaluation mode.
        self.eval()

        # loop over dataset.
        for x, y in dataset:
            self.zero_grad()
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            output = self(x.view(1,-1))
            #-------------------------------------------------------------------------#
            ## OPTION 1: use predicted label to calculate loglikelihood:
            label = output.max(1)[1]
            ## OPTION 2: use true label to calculate loglikelihood:
            #label = torch.LongTensor([y]) if type(y) == int else y
            #label = Variable(label).cuda() if self._is_on_cuda() else Variable(label)
            ##--> based on the permuted MNIST task, OPTION 1 seems to work better
            #-------------------------------------------------------------------------#
            if int(torch.__version__[2])>2:
                loglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            else:
                loglikelihood = F.nll_loss(F.log_softmax(output), label)
            loglikelihood.backward()
            for n, p in self.named_parameters():
                n = n.replace('.', '__')
                est_fisher_info[n] += p.grad.data ** 2
                ## QUESTION: assumption here is that the mean of the gradient at the
                ##           evaluated point is zero, but in practice this doesn't hold.
                ##           Should this be corrected for? Currently parameters with a
                ##           large average gradient left are thought to be important
                ##           (which to some degree actually makes sense, since this
                ##            indicates that changing these parameters will have a
                ##            large effect on performance)

        # normalize by sample size used for estimation
        est_fisher_info = {n: p / len(dataset) for n, p in est_fisher_info.items()}

        # consolidate new values in the network
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_prev_task'.format(n), p.data.clone())
            self.register_buffer('{}_estimated_fisher'.format(n), est_fisher_info[n])

    def ewc_loss(self, lamda, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_prev_task'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there is no consolidated "estimated_fisher".
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )


    #------------- "Intelligent Synapses"-specifc functions -------------#

    def update_omega(self, W, epsilon):
        '''After each task.'''
        for n, p in self.named_parameters():
            n = n.replace('.', '__')

            # calculate new values
            p_prev = getattr(self, '{}_prev_task'.format(n))
            p_current = p.data.clone()
            p_change = p_current - p_prev
            omega_add = W[n]/(p_change**2 + epsilon)
            try:
                omega = getattr(self, '{}_omega'.format(n))
            except AttributeError:
                omega = p.data.clone().zero_()
            omega_new = omega + omega_add

            # consolidate new values in the network
            self.register_buffer('{}_prev_task'.format(n), p_current)
            self.register_buffer('{}_omega'.format(n), omega_new)

    def surrogate_loss(self, c, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve previous parameter values and their normalized path integral (i.e., omega).
                n = n.replace('.', '__')
                prev_values = getattr(self, '{}_prev_task'.format(n))
                omega = getattr(self, '{}_omega'.format(n))
                # wrap them in variables.
                prev_values = Variable(prev_values)
                omega = Variable(omega)
                # calculate the surrogate loss, sum over all parameters
                losses.append((omega * (p-prev_values)**2).sum())
            return c*sum(losses)
        except AttributeError:
            # surrogate loss is 0 if there is no consolidated "omega".
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )
