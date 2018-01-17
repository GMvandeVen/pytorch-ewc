from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual_visdom
import visual_plt
from matplotlib.backends.backend_pdf import PdfPages
import random

def train(model, train_datasets, test_datasets, epochs_per_task=10,
          batch_size=64, test_size=1024, consolidate=True,
          fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5, lamda=3,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False,
          plot="pdf",
          pdf_file_name=None):

    # number of tasks
    n_tasks = len(train_datasets)

    # prepare the loss criterion and the optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # if plotting, prepare task names and plot-titles
    if not plot=="none":
        names = ['task {}'.format(i + 1) for i in range(n_tasks)]
        title_precision = 'precision (consolidated)' if consolidate else 'precision'
        title_loss = 'loss (consolidated)' if consolidate else 'loss'

    # if plotting in pdf, initiate lists for storing data
    if plot=="pdf":
        all_task_lists = [[] for _ in range(n_tasks)]
        x_list = []
        all_loss_lists = [[] for _ in range(3)]
        x_loss_list = []

    # training, ..looping over all tasks
    for task, train_dataset in enumerate(train_datasets, 1):

        # ..looping over all epochs
        for epoch in range(1, epochs_per_task+1):

            # prepare data-loader, and wrap in "tqdm"-object.
            data_loader = utils.get_data_loader(
                train_dataset, batch_size=batch_size, cuda=cuda
            )
            data_stream = tqdm(enumerate(data_loader, 1))

            # ..looping over all batches
            for batch_index, (x, y) in data_stream:

                # where are we?
                data_size = len(x)
                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
                previous_task_iteration = sum([
                    epochs_per_task * len(d) // batch_size for d in
                    train_datasets[:task-1]
                ])
                current_task_iteration = (epoch-1)*dataset_batches + batch_index
                iteration = previous_task_iteration + current_task_iteration

                # prepare the data.
                x = x.view(data_size, -1)
                x = Variable(x).cuda() if cuda else Variable(x)
                y = Variable(y).cuda() if cuda else Variable(y)

                # run model, backpropagate errors, update parameters.
                model.train()
                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criterion(scores, y)
                ewc_loss = model.ewc_loss(lamda, cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

                # calculate the training precision.
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().data[0] / len(x)

                # print progress to the screen using "tqdm"
                data_stream.set_description((
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4}'
                ).format(
                    task=task,
                    tasks=n_tasks,
                    epoch=epoch,
                    epochs=epochs_per_task,
                    trained=batch_index*batch_size,
                    total=dataset_size,
                    progress=(100.*batch_index/dataset_batches),
                    prec=precision,
                    ce_loss=ce_loss.data[0],
                    ewc_loss=ewc_loss.data[0],
                    loss=loss.data[0],
                ))

                # Send test precision to the visdom server,
                #  or store for later plotting to pdf.
                if not plot=="none":
                    if iteration % eval_log_interval == 0:
                        precs = [
                            utils.validate(
                                model, test_datasets[i], test_size=test_size,
                                cuda=cuda, verbose=False,
                            ) if i+1 <= task else 0 for i in range(n_tasks)
                        ]
                        if plot=="visdom":
                            visual_visdom.visualize_scalars(
                                precs, names, title_precision,
                                iteration, env=model.name,
                            )
                        elif plot=="pdf":
                            for task_id, _ in enumerate(names):
                                all_task_lists[task_id].append(precs[task_id])
                            x_list.append(iteration)

                # Send losses to the visdom server,
                #  or store for later plotting to pdf.
                if not plot=="none":
                    if iteration % loss_log_interval == 0:
                        if plot=="visdom":
                            visual_visdom.visualize_scalars(
                                [loss.data, ce_loss.data, ewc_loss.data],
                                ['total', 'cross entropy', 'ewc'],
                                title_loss, iteration, env=model.name
                            )
                        elif plot=="pdf":
                            all_loss_lists[0].append(loss.data.cpu().numpy()[0])
                            all_loss_lists[1].append(ce_loss.data.cpu().numpy()[0])
                            all_loss_lists[2].append(ewc_loss.data.cpu().numpy()[0])
                            x_loss_list.append(iteration)

        if consolidate:
            # take random samples from every task learned so far
            dataset_old_tasks = []
            for prev_task_id in range(task):
                prev_dataset = train_datasets[prev_task_id]
                sample_ids = random.sample(
                    range(len(prev_dataset)), fisher_estimation_sample_size
                )
                sample = [prev_dataset[id] for id in sample_ids]
                dataset_old_tasks += sample
            # from all those samples, randomly select [fisher_estimation_sample_size]
            dataset_old_tasks = random.sample(dataset_old_tasks, k = fisher_estimation_sample_size)
            # estimate the Fisher Information matrix and consolidate it in the network
            model.consolidate(model.estimate_fisher(dataset_old_tasks))

            # after each task, estimate fisher information of the parameters
            #  and consolidate them in the network.
            # model.consolidate(model.estimate_fisher_old(
            #     train_dataset, fisher_estimation_sample_size
            # ))

    # if requested, generate pdf.
    if plot=="pdf":
        # create list to store all figures to be plotted.
        figure_list = []

        # Fig1: precision
        figure = visual_plt.plot_lines(
            all_task_lists, x_axes=x_list, line_names=names
        )
        figure_list.append(figure)

        # Fig2: loss
        figure = visual_plt.plot_lines(
            all_loss_lists, x_axes=x_loss_list,
            line_names=['total', 'cross entropy', 'ewc']
        )
        figure_list.append(figure)

        # create pdf containing all figures.
        pdf = PdfPages(pdf_file_name)
        for figure in figure_list:
            pdf.savefig(figure)
        pdf.close()