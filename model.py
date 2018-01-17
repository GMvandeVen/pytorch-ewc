from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils


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
        '''Estimates diagonal of parameter Fisher Information matrix based on entries in [dataset].

        [dataset]:  list of data-points [x, y] (should be random sample from all previous tasks)'''

        # prepare dictionary to store estimated Fisher Information matrix.
        est_fisher_info = {}
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            est_fisher_info[n] = Variable(p.data.clone().zero_())

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
            #-------------------------------------------------------------------------#
            if int(torch.__version__[2])>2:
                loglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            else:
                loglikelihood = F.nll_loss(F.log_softmax(output), label)
            loglikelihood.backward()
            for n, p in self.named_parameters():
                n = n.replace('.', '__')
                est_fisher_info[n].data += p.grad.data ** 2
                ## QUESTION: assumption here is that the mean of the gradient at the
                ##           evaluated point is zero, but in practice this doesn't hold.
                ##           Should this be corrected for? Currently parameters with a
                ##           large average gradient left are thought to be important
                ##           (which to some degree actually makes sense, since this
                ##            indicates that changing these parameters will have a
                ##            large effect on performance)

        # normalize by sample size used for estimation.
        est_fisher_info = {n: p / len(dataset) for n, p in est_fisher_info.items()}

        return est_fisher_info

    def estimate_fisher_old(self, dataset, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        data_loader = utils.get_data_loader(dataset, batch_size)
        loglikelihoods = []
        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                # should this be evaluated at the true values or the predicted ones?
                F.log_softmax(self(x))[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters())
        parameter_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            self.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, lamda, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_estimated_mean'.format(n))
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
            # ewc loss is 0 if there are no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )


    #------------- "Intelligent Synapses"-specifc functions -------------#

    # def update_importance_estimate(self):
    #     '''After each parameter-update.'''
    #
    # def update_omega(self):
    #     '''After each task.'''
    #     for n, p in self.named_parameters():
    #         n = n.replace('.', '__')
    #         self.register_buffer('{}_prev_value'.format(n), p.data.clone())
    #
    #         # Add to the new integral
    #         getattr(self, '{}_norm_intergral'.format(n)).add_()
    #
    #         self.register_buffer('{}_norm_integral'.format(n),
    #                              fisher[n].data.clone())

    def surrogate_loss(self, c, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve previous parameter values and their normalized path integral.
                n = n.replace('.', '__')
                prev_values = getattr(self, '{}_prev_value'.format(n))
                norm_integral = getattr(self, '{}_norm_intergral'.format(n))
                # wrap them in variables.
                prev_values = Variable(prev_values)
                norm_integral = Variable(norm_integral)
                # calculate the surrogate loss, sum over all parameters
                losses.append((norm_integral * (p-prev_values)**2).sum())
            return c*sum(losses)
        except AttributeError:
            # surrogate loss is 0 if there are no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )
