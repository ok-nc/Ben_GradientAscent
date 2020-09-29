"""
Wrapper functions for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
#from torchsummary import summary
from torch.optim import lr_scheduler
#from torchviz import make_dot
#from network_model import Lorentz_layer
from plotting_functions import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_Lor_params

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelmax



class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The network architecture object
        self.flags = flags                                      # The flags containing the hyperparameters
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # Network training mode, create a new ckpt folder
            if flags.model_name is None:                    # Use custom name if possible, otherwise timestamp
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_custom_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train()
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train()
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for tensorboard
        self.best_validation_loss = 0.1    # Set the BVL to large number
        self.best_pretrain_loss = float('inf')
        self.running_loss = []
        self.pre_train_model = self.flags.pre_train_model

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_data=(8,))
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('There are %d trainable out of %d total parameters' %(pytorch_total_params, pytorch_total_params_train))
        return model

    def make_MSE_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistent both at training time \
        and inference time for a backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss of the network
        return MSE_loss

    def mirror_padding(self, input, pad_width=1):
        """
        pads the input tensor by mirroring
        :param input: The tensor to be padded
        :param pad_width: The padding width (default to be 1)
        :return: the padded tensor
        """
        # Get the shape and create new tensor
        shape = np.array(np.shape(input.detach().cpu().numpy()))
        shape[-1] += 2*pad_width
        padded_tensor = torch.zeros(size=tuple(shape))
        #print("shape in mirror: ", np.shape(padded_tensor))
        padded_tensor[:, pad_width:-pad_width] = input
        padded_tensor[:, 0:pad_width] = input[:, pad_width:2*pad_width]
        padded_tensor[:, -pad_width:] = input[:, -2*pad_width:-pad_width]
        if torch.cuda.is_available():
            padded_tensor = padded_tensor.cuda()
        return padded_tensor

    # Peak finder loss
    def peak_finder_loss(self, logit=None, labels=None, w0=None, w_base=None):
        batch_size = labels.size()[0]
        #batch_size = 1
        loss_penalty = 100
        # Define the convolution window for peak finding
        descend = torch.tensor([0, 1, -1], requires_grad=False, dtype=torch.float32)
        ascend = torch.tensor([-1, 1, 0], requires_grad=False, dtype=torch.float32)
        # Use GPU option
        if torch.cuda.is_available():
            ascend = ascend.cuda()
            descend = descend.cuda()
        # make reflection padding
        padded_labels = self.mirror_padding(labels)
        # Get the maximum and minimum values
        max_values = F.conv1d(padded_labels.view(batch_size, 1, -1),
                             ascend.view(1, 1, -1), bias=None, stride=1)
        min_values = F.conv1d(padded_labels.view(batch_size, 1, -1),
                                     descend.view(1, 1, -1), bias=None, stride=1)
        #print("shape of minvalues", np.shape(min_values))
        # Special arrangement for peaks at the edge
        #print("Length of min=0", np.sum(min_values.detach().cpu().numpy() == 0))
        #print("Length of max=0", np.sum(max_values.detach().cpu().numpy() == 0))
        #min_values[min_values == 0] = 1
        min_values = F.relu(min_values)
        #max_values[max_values == 0] = 1
        max_values = F.relu(max_values)

        # Get the peaks
        zeros = torch.mul(max_values, min_values).squeeze() > 0
        """
        # Debugging code for the peaks
        float_zeros = zeros.float().cpu().numpy()
        sums = np.sum(float_zeros, axis=1)
        print("The ones that are not having 4 peaks")
        print("shape of sums ", np.shape(sums))
        print("number of spectra less than 4 peaks", np.sum(sums!=4))
        print(sums)
        print(np.arange(batch_size)[sums != 4])
        print("sums of peaks", np.sum(sums))
        print("shape of zeros ", np.shape(zeros))
        w_base_expand = w_base.expand_as(zeros)
        print("shape of the w_base_expand", np.shape(w_base_expand))
        ###############################
        # Plotting to see the spectra #
        ###############################
        plot_num = 77
        f = plt.figure()
        plot_w = self.model.w.cpu().numpy()
        plot_spectra = labels.detach().cpu().numpy()[plot_num, :]
        #print("shape of wplot", np.shape(plot_w))
        #print("shape of spectra", np.shape(plot_spectra))
        plt.plot(plot_w, plot_spectra )
        plt.savefig('{}.jpg'.format(plot_num))
        """
        ###############################
        ###############################
        peaks = torch.zeros(size=[batch_size, 4], requires_grad=False, dtype=torch.float32)
        for i in range(batch_size):
            peak_current = w_base[zeros[i, :]]
            #print("len of peak_current: ", len(peak_current))
            peak_num = len(peak_current)
            if peak_num == 4:
                peaks[i, :] = peak_current
            else:
                peak_rank, index = torch.sort(labels[i, zeros[i, :]])  # Get the rank of the peaks
                peaks[i, :peak_num] = peak_current                  # put the peaks into first len spots
                peaks[i, peak_num:] = peak_current[index[0]]        # make the full array using the highest peak
        #peaks = torch.tensor(w_base_expand[zeros], requires_grad=False, dtype=torch.float32)
        if torch.cuda.is_available():
            peaks = peaks.cuda()
        # sort the w0 to match the peak orders
        w0_sort, indices = torch.sort(w0)
        #print("shape of w0_sort ", np.shape(w0_sort))
        #print("shape of peaks ", np.shape(peaks))
        #print(w0_sort)
        #print(peaks)
        return loss_penalty * F.mse_loss(w0_sort, peaks)

    def make_custom_loss(self, logit=None, labels=None, w0=None,
                         g=None, wp=None, epoch=None, peak_loss=False,
                         gt_lor=None, lor_ratio=0, lor_weight=1,
                         lor_loss_only=False, gt_match_style='undefined',
                         gradient_descend=True):
        """
        The custom master loss function
        :param logit: The model output
        :param labels: The true label
        :param w0: The Lorentzian parameter output w0
        :param g: The Lorentzian parameter output g
        :param wp: The Lorentzian parameter output wp
        :param epoch: The current epoch number
        :param peak_loss: Whether use peak_finding_loss or not
        :param gt_lor: The ground truth Lorentzian parameter
        :param lor_ratio: The ratio of lorentzian parameter to spectra during training
        :param lor_loss_only: The flag to have only lorentzian loss, this is for alternative training
        :param gt_match_style: The style of matching the GT Lor param to the network output:
              'gt': The ground truth correspondence matching
              'random': The permutated matching in random
              'peak': Match according to the sorted peaks (w0 values sequence)
        :param gradient_descend: The flag of gradient descend or ascend
        :return:
        """
        if logit is None:
            return None

        ############
        # MSE Loss #
        ############
        custom_loss = nn.functional.mse_loss(logit, labels, reduction='mean')

        # Gradient ascent
        if gradient_descend is False:
            custom_loss *= -self.flags.gradient_ascend_strength

        ######################
        # Boundary loss part #
        ######################
        if w0 is not None:
            freq_mean = (self.flags.freq_low + self.flags.freq_high)/ 2
            freq_range = (self.flags.freq_high - self.flags.freq_low)/ 2
            custom_loss += torch.sum(torch.relu(torch.abs(w0 - freq_mean) - freq_range))
        if g is not None:
            if epoch is not None and epoch < 100:
                custom_loss += torch.sum(torch.relu(-g + 0.05))
            else:
                custom_loss += 100 * torch.sum(torch.relu(-g))
        if wp is not None:
            custom_loss += 100*torch.sum(torch.relu(-wp))

        #####################
        # Peak finding loss #
        #####################
        if peak_loss and epoch < 100:
            custom_loss += self.peak_finder_loss(labels=labels, w0=w0, w_base=self.model.w)

        ###############################
        # Facilitated Lorentzian loss #
        ###############################
        if gt_lor is not None and epoch < 250:
            # Probablistic accepting Lorentzian loss
            random_number = np.random.uniform(size=1)
            if random_number < lor_ratio:   # Accept this with Lorentzian loss
                # 2020.07.12 to test whether the permutated Lorentzian label can have the same effect
                if gt_match_style == 'random':
                    permuted = np.random.permutation(4)
                    lor_loss = nn.functional.mse_loss(w0, gt_lor[:, permuted])
                    lor_loss += nn.functional.mse_loss(wp, gt_lor[:, permuted + 4])
                    lor_loss += nn.functional.mse_loss(g, gt_lor[:, permuted + 8] * 10)
                elif gt_match_style == 'gt':
                    lor_loss = nn.functional.mse_loss(w0, gt_lor[:, :4])
                    lor_loss += nn.functional.mse_loss(wp, gt_lor[:, 4:8])
                    lor_loss += nn.functional.mse_loss(g, gt_lor[:, 8:12]*10)
                elif gt_match_style == 'peak':
                    # Sort the gt and network output
                    gt_w0_sort, gt_rank_index = torch.sort(gt_lor[:, :4])
                    ntwk_w0_sort, ntwk_rank_index = torch.sort(w0)
                    lor_loss = nn.functional.mse_loss(w0[:, ntwk_rank_index], gt_lor[:, gt_rank_index])
                    lor_loss += nn.functional.mse_loss(wp[:, ntwk_rank_index], gt_lor[:, gt_rank_index+4])
                    lor_loss += nn.functional.mse_loss(g[:, ntwk_rank_index], gt_lor[:, gt_rank_index+8] * 10)
                else:
                    raise ValueError("Your gt_match_style in custom loss should be one of 'random',"
                                     " 'gt' or 'peak', please contact Ben for details")

                if lor_loss_only:       # Alternating training activated
                    custom_loss = lor_loss
                else:                   # Combined loss
                    custom_loss += lor_loss * lor_weight


        return custom_loss


    def make_optimizer(self, param=None):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if param is None:
            param = self.model.parameters()
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'AdamW':
            op = torch.optim.AdamW(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'Adamax':
            op = torch.optim.Adamax(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SparseAdam':
            op = torch.optim.SparseAdam(param, lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif self.flags.optim == 'LBFGS':
            op = torch.optim.LBFGS(param, lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        # return lr_scheduler.StepLR(optimizer=self.optm, step_size=50, gamma=0.75, last_epoch=-1)
        try:
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm_all, mode='min',
                                        factor=self.flags.lr_decay_rate,
                                          patience=10, verbose=True, threshold=1e-4)
        except:
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                                  factor=self.flags.lr_decay_rate,
                                                  patience=10, verbose=True, threshold=1e-4)



    def save(self):
        """
        Saving the model to the current check point folder with name best_model_MSE3.pt
        :return: None
        """
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_MSE3.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_MSE3.pt
        :return:
        """
        self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_MSE3.pt'))

    def load_pretrain(self):
        """
        Loading the model from the check point folder with name best_model_MSE3.pt
        :return:
        """
        try:
            pt_file = os.path.join('models', self.pre_train_model, 'best_pretrained_model.pt')
            self.model = torch.load(pt_file)
        except:
            pt_file = os.path.join('models', self.pre_train_model, 'best_model_MSE3.pt')
            self.model = torch.load(pt_file)


    def record_weight(self, name='Weights', layer=None, batch=999, epoch=999):
        """
        Record the weights for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0:
            weights = layer.weight.cpu().data.numpy()   # Get the weights

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(weights.shape))
            wmax = np.amax(np.asarray(weights.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq**2 - (wmin * wmax))), dtype='float64')
            weights = weights.reshape((1, -1))
            weights = np.concatenate((weights, diff), axis=1)
            # f = plt.figure(figsize=(10, 5))
            # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
            # plt.colorbar(c, fraction=0.03)
            f = plot_weights_3D(weights.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Weights_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def record_grad(self, name='Gradients', layer=None, batch=999, epoch=999):
        """
        Record the gradients for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0 and epoch > 0:
            gradients = layer.weight.grad.cpu().data.numpy()

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(gradients.shape))
            wmax = np.amax(np.asarray(gradients.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq ** 2 - (wmin * wmax))), dtype='float64')
            gradients = gradients.reshape((1, -1))
            gradients = np.concatenate((gradients, diff), axis=1)
            # f = plt.figure(figsize=(10, 5))
            # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
            # plt.colorbar(c, fraction=0.03)
            f = plot_weights_3D(gradients.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Gradients_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def reset_lr(self, optm):
        """
        Reset the learning rate to to original lr
        :param optm: The optimizer
        :return: None
        """
        self.lr_scheduler = self.make_lr_scheduler()
        for g in optm.param_groups:
            g['lr'] = self.flags.lr

    def train_stuck_by_lr(self, optm, lr_limit):
        """
        Detect whether the training is stuck with the help of LR scheduler which decay when plautue
        :param optm: The optimizer
        :param lr_limit: The limit it judge it is stuck
        :return: Boolean value of whether it is stuck
        """
        for g in optm.param_groups:
            if g['lr'] < lr_limit:
                return True
            else:
                return False


    def train(self):
        """
        The major training function. This starts the training using parameters given in the flags
        :return: None
        """
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm_all = self.make_optimizer()
        # Separate optimizer for w0 since it is very important
        self.optm_w0 = self.make_optimizer(param=[params for params in self.model.linears.parameters()]\
                                           .append(self.model.lin_w0.parameters()))
        self.lr_scheduler = self.make_lr_scheduler()

        # Set the training flag to be True at the start, this is for the gradient ascend
        train_flag = True
        epoch = 0
        gradient_descend = True

        #for epoch in range(self.flags.train_step):         # Normal training
        while train_flag:
            if gradient_descend is False:
                print('This is Epoch {} doing gradient ascend to avoid local minimum'.format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU

                self.optm_all.zero_grad()                                   # Zero the gradient first
                #print("mean of spectra target", np.mean(spectra.data.numpy()))
                logit,w0,wp,g = self.model(geometry)            # Get the output

                ################
                # Facilitation #
                ################
                #loss = self.make_custom_loss(logit, spectra[:, 12:], w0=w0,
                #                             g=g, wp=wp, epoch=epoch, peak_loss=False,
                #                            gt_lor=spectra[:, :12], lor_ratio=self.flags.lor_ratio,
                #                            lor_weight=self.flags.lor_weight, gt_match_style='gt')
                loss = self.make_custom_loss(logit, spectra, gradient_descend=gradient_descend,
                                             w0=w0, g=g, wp=wp)
                # print(loss)
                loss.backward()
                self.optm_all.step()  # Move one step the optimizer
                if epoch % self.flags.record_step == 0:
                    if j == 0:
                        for k in range(self.flags.num_plot_compare):
                            f = compare_spectra(Ypred=logit[k, :].cpu().data.numpy(),
                                                Ytruth=spectra[k, :].cpu().data.numpy(),
                                                xmin=self.flags.freq_low,
                                                xmax=self.flags.freq_high, num_points=self.flags.num_spec_points)
                            self.log.add_figure(tag='Test ' + str(k) + ') Sample T Spectrum'.format(1),
                                                figure=f, global_step=epoch)
                """
                if self.flags.use_clip:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)
    
                if epoch % self.flags.record_step == 0:
                    if j == 0:
                        for k in range(self.flags.num_plot_compare):
                            f = compare_spectra(Ypred=logit[k, :].cpu().data.numpy(),
                                                     Ytruth=spectra[k, 12:].cpu().data.numpy(), E2=self.model.e2[k,:,:], xmin=self.flags.freq_low,
                                                xmax=self.flags.freq_high, num_points=self.flags.num_spec_points)
                            self.log.add_figure(tag='Test ' + str(k) +') Sample e2 Spectrum'.format(1),
                                                figure=f, global_step=epoch)

                """

                """
                ####################################
                # Extra training loop for w0 alone #
                ####################################
                for ii in range(self.flags.optimize_w0_ratio):
                    self.optm_w0.zero_grad()
                    logit, w0, wp, g = self.model(geometry)  # Get the output
                    loss = self.make_custom_loss(logit, spectra[:, 12:], w0=w0, g=g, wp=wp,
                                                 epoch=ii, peak_loss=True)
                    loss.backward()
                    self.optm_w0.step()
                
                ######################################
                # Alternating training for Lor param #
                ######################################
                if np.random.uniform(size=1) < self.flags.lor_ratio and False:
                    print("entering one batch of facilitated training")
                    for ii in range(self.flags.train_lor_step):
                        self.optm_all.zero_grad()
                        logit, w0, wp, g = self.model(geometry)  # Get the output
                        loss = self.make_custom_loss(logit, spectra[:, 12:], w0=w0, g=g, wp=wp,
                                                     epoch=ii, peak_loss=True, gt_lor=spectra[:, :12],
                                                     lor_ratio=1,
                                                     lor_weight=self.flags.lor_weight,
                                                     lor_loss_only=True)
                        loss.backward()
                        self.optm_all.step()
                """
                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss
                self.running_loss.append(np.copy(loss.cpu().data.numpy()))


            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            if not gradient_descend:
                train_avg_loss *= -1
            #train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

            # Validation part
            if epoch % self.flags.eval_step == 0 or not gradient_descend:           # For eval steps, do the evaluations and tensor board
                # Record the training loss to tensorboard
                #train_avg_loss = train_loss.data.numpy() / (j+1)
                self.log.add_scalar('Loss/ Training Loss', train_avg_loss, epoch)
                #self.log.add_scalar('Loss/ Batchnorm Training Loss', train_avg_eval_mode_loss, epoch)
                # self.log.add_scalar('Running Loss', train_avg_eval_mode_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit,w0,wp,g = self.model(geometry)
                        #loss = self.make_MSE_loss(logit, spectra)                   # compute the loss
                        loss = self.make_custom_loss(logit, spectra)#, w0=w0, g=g, wp=wp,
                                                     #epoch=epoch, peak_loss=True)
                        test_loss.append(np.copy(loss.cpu().data.numpy()))           # Aggregate the loss

                        """
                        if j == 0 and epoch % self.flags.record_step == 0:
                            # f2 = plotMSELossDistrib(test_loss)
                            f2 = plotMSELossDistrib(logit.cpu().data.numpy(), spectra[:, 12:].cpu().data.numpy())
                            self.log.add_figure(tag='0_Testing Loss Histogram'.format(1), figure=f2,
                                                global_step=epoch)
                        """
                # Record the testing loss to the tensorboard

                test_avg_loss = np.mean(test_loss)
                self.log.add_scalar('Loss/ Validation Loss', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        return None

            if gradient_descend:                # If currently in gradient descend mode
                # # Learning rate decay upon plateau
                self.lr_scheduler.step(train_avg_loss)
                if loss.detach().cpu().numpy() > 0.01:
                    # If the LR changed (i.e. training stuck) and also loss is large
                    if self.train_stuck_by_lr(self.optm_all, self.flags.lr/8):
                        # Switch to the gradient ascend mode
                        gradient_descend = False
                else:
                    print("The loss is lower than 0.01! good news")
                    # Stop the training
                    train_flag = False
                    #self.save()
                    #print("Saving the model...")
            else:                               # Currently in ascent mode, change to gradient descend mode
                print("After the gradient ascend, switching back to gradient descend")
                gradient_descend = True         # Change to Gradient descend mode
                self.reset_lr(self.optm_all)    # reset lr

            epoch += 1
            if epoch > self.flags.train_step:
                train_flag = False

            """
            ################
            # Warm restart #
            ################
            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.optm.param_groups:
                        param_group['lr'] = self.flags.lr
                        print('Resetting learning rate to %.5f' % self.flags.lr)
            """

        self.log.close()
        # np.savetxt(time.strftime('%Y%m%d_%H%M%S', time.localtime())+'.csv', self.running_loss, delimiter=",")
        #self.save()

    def pretrain(self):
        """
        The pretraining function. This starts the pretraining using parameters given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-3)
        self.lr_scheduler = self.make_lr_scheduler()

        # Start a tensorboard session for logging loss and training images
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()

        print("Starting pre-training process")
        pre_train_epoch = 300
        for epoch in range(pre_train_epoch):
            # print("This is pretrainin Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            sim_loss_list = []

            self.model.train()
            for j, (geometry, params_truth) in enumerate(self.train_loader):
                # if j == 0 and epoch == 0:
                    # print(geometry)

                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    params_truth = params_truth.cuda()  # Put data onto GPU
                self.optm.zero_grad()  # Zero the gradient first
                logit,w0,wp,g = self.model(geometry)  # Get the output
                # print("label size:", params_truth.size())
                # print("logit size:", params.size())

                pretrain_loss = self.make_MSE_loss(w0, params_truth[:, :4])  # Get the loss tensor
                pretrain_loss += self.make_MSE_loss(wp, params_truth[:, 4:8])  # Get the loss tensor
                pretrain_loss += self.make_MSE_loss(g, params_truth[:, 8:12]*10)  # Get the loss tensor
                sim_loss = self.make_MSE_loss(logit, params_truth[:, 12:])
                pretrain_loss.backward()  # Calculate the backward gradients
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
                train_loss.append(np.copy(pretrain_loss.cpu().data.numpy()))  # Aggregate the loss
                sim_loss_list.append(np.copy(sim_loss.cpu().data.numpy()))
                if epoch < pre_train_epoch - 1:
                    self.optm.step()  # Move one step the optimizer

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            sim_loss = np.mean(sim_loss_list)
            self.running_loss.append(sim_loss)

            if epoch % 20 == 0:  # Evaluate every 20 steps
                # Record the training loss to the tensorboard
                # train_avg_loss = train_loss.data.numpy() / (j+1)
                self.log.add_scalar('Pretrain Loss', train_avg_loss, epoch)
                self.log.add_scalar('Simulation Loss', sim_loss, epoch)

                for j in range(self.flags.num_plot_compare):
                    f = compare_Lor_params(w0=w0[j, :].cpu().data.numpy(), wp=wp[j, :].cpu().data.numpy(),
                                           g=g[j, :].cpu().data.numpy(),
                                           truth=params_truth[j, :12].cpu().data.numpy())
                    self.log.add_figure(tag='Test ' + str(j) + ') e2 Lorentz Parameter Prediction'.
                                        format(1), figure=f, global_step=epoch)

                # Pretraining files contain both Lorentz parameters and simulated model spectra
                pretrain_sim_prediction = params_truth[:, 12:]
                pretrain_model_prediction = logit
                #pretrain_model_prediction = Lorentz_layer(w0,wp,g/10)

                for j in range(self.flags.num_plot_compare):
                    f = compare_spectra(Ypred=pretrain_model_prediction[j, :].cpu().data.numpy(),
                                             Ytruth=pretrain_sim_prediction[j, :].cpu().data.numpy())
                    self.log.add_figure(tag='Test ' + str(j) + ') e2 Model Prediction'.format(1),
                                        figure=f, global_step=epoch)

                print("This is Epoch %d, pretrain loss %.5f,, and sim loss is %.5f" % (
                epoch, train_avg_loss,  sim_loss))

                # Model improving, save the model
                if train_avg_loss < self.best_pretrain_loss:
                    self.best_pretrain_loss = train_avg_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_pretrain_loss < self.flags.stop_threshold:
                        print("Pretraining finished EARLIER at epoch %d, reaching loss of %.5f" % \
                              (epoch, self.best_pretrain_loss))
                        return None

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)

            # Save pretrained model at end
            if epoch == pre_train_epoch-1:
                # weights = self.model.linears[-1].weight.cpu().data.numpy()
                # # print(weights.shape)
                # np.savetxt('Pretrain_Lorentz_Weights.csv', weights, fmt='%.3f', delimiter=',')
                torch.save(self.model, os.path.join(self.ckpt_dir, 'best_pretrained_model.pt'))
                # self.record_weight(name='Pretraining', batch=0, epoch=999)

        self.log.close()


    def evaluate(self, save_dir='eval/'):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()                       # Evaluation mode

        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(self.saved_model))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(self.saved_model))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(self.saved_model))
        # Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(self.saved_model))  # For pure forward model, there is no Xpred

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt, open(Ypred_file, 'a') as fyp:
            # Loop through the eval data and evaluate
            with torch.no_grad():
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit,w0,wp,g = self.model(geometry)  # Get the output
                    np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyt, spectra.cpu().data.numpy()[:, 12:], fmt='%.3f')
                    np.savetxt(fyp, logit.cpu().data.numpy(), fmt='%.3f')
        return Ypred_file, Ytruth_file






