from copy import copy
import imp
import logging

import torch
# from torch import nn
import torch.nn as nn

from fedml_api.model.SNIP.snip import SNIP

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer

import json
import numpy as np



class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def apply_network_masks(self, keep_masks):
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), self.model.modules())

        for layer, keep_mask in zip(prunable_layers, keep_masks):
            assert (layer.weight.shape == keep_mask.shape)

            layer.weight.data[keep_mask == 0.] = 0.
        # prunable_layers = filter(
        #     lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
        #         layer, nn.Linear), self.model.modules())

        # sum_zeros = 0
        # for k, keep_masks in keep_masks_dict.items():
        #     logging.info(f'prun masks from {k}')
        #     for layer, keep_mask in zip(prunable_layers, keep_masks):
        #         assert (layer.weight.shape == keep_mask.shape)

        #         layer.weight.data[keep_mask == 0.] = 0.
        #         #  sum(np.where(diff, 0, 1))
        #         sum_zeros += sum(np.where(keep_mask.view(-1).numpy(), 0, 1))
        # logging.info(f'^^^^^^^^^^^^^^^^sum_zero: {sum_zeros}')
        # pass

    def get_model_gradient(self):
        finish_params = self.get_model_params()
        # count = 0 
        for param_tensor in self.init_params:
            finish_params[param_tensor] -= self.init_params[param_tensor]
            # self.init_params[param_tensor] -= finish_params[param_tensor]
            # diff_params = init_params[param_tensor] - finish_params[param_tensor]
            # diff_params = diff_params.view(-1).numpy()
            # print(diff_params)
            # count += sum(np.where(diff_params, 0, 1))
            # logging.info(sum(np.where(diff_params, 0, 1)))
        #     gradient = self.init_params[param_tensor].view(-1).numpy()
        #     count += sum(np.where(gradient, 0, 1))
        # logging.info(f'########gradient: {count}')
        return finish_params

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def apply_prune_mask(self, net, keep_masks):
        # Before I can zip() layers and pruning masks I need to make sure they match
        # one-to-one by removing all the irrelevant modules:
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), net.modules())
        handles = []

        for layer, keep_mask in zip(prunable_layers, keep_masks):
            assert (layer.weight.shape == keep_mask.shape)

            def hook_factory(keep_mask):
                """
                The hook function can't be defined directly here because of Python's
                late binding which would result in all hooks getting the very last
                mask! Getting it through another function forces early binding.
                """

                def hook(grads):
                    # return grads * keep_mask
                    return grads * keep_mask

                return hook

            # mask[i] == 0 --> Prune parameter
            # mask[i] == 1 --> Keep parameter

            # Step 1: Set the masked weights to zero (NB the biases are ignored)
            # Step 2: Make sure their gradients remain zero
            layer.weight.data[keep_mask == 0.] = 0.
            # layer.weight.register_hook(hook_factory(keep_mask))
            handles.append(layer.weight.register_hook(hook_factory(keep_mask)))
        return handles

    # def rebuild_net(self, net, keep_masks):
    #     prunable_layers = filter(
    #         lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
    #             layer, nn.Linear), net.modules())

    #     for layer, keep_mask in zip(prunable_layers, keep_masks):
    #         assert (layer.weight.shape == keep_mask.shape)

    #         def hook_factory(keep_mask):
    #             """
    #             The hook function can't be defined directly here because of Python's
    #             late binding which would result in all hooks getting the very last
    #             mask! Getting it through another function forces early binding.
    #             """

    #             def hook(grads):

    #                 return grads

    #             return hook

    #         # mask[i] == 0 --> Prune parameter
    #         # mask[i] == 1 --> Keep parameter

    #         # Step 1: Set the masked weights to zero (NB the biases are ignored)
    #         # Step 2: Make sure their gradients remain zero
    #         # layer.weight.data[keep_mask == 0.] = 0.
    #         layer.weight.register_hook(hook_factory(keep_mask))

    def record_keep_masks(self, keep_masks):
        masks = torch.cat([torch.flatten(x) for x in keep_masks]).to('cpu').tolist()
        with open(f'./{self.id}_keep_masks.txt', 'a+') as f:
            f.write(json.dumps(masks))
            f.write('\n')

    def train(self, train_data, device, args):
        # return
        
        model = self.model
        model.to(device)

        handles = None
        # if self.keep_masks == None: 
        self.keep_masks = SNIP(self.model, 0.1, train_data, device)
        handles = self.apply_prune_mask(self.model, self.keep_masks)
        # self.record_keep_masks(self.keep_masks)
        
        self.init_params = self.get_model_params()
        model.to(device)
        # if args.only_1r_prun and (self.keep_masks == None):
        #     logging.info(f'client {self.id} pruning network')
        #     # self.keep_masks = SNIP(model, 0.1, train_data, device)
        #     self.keep_masks = SNIP(self.model, 0.1, train_data, device)
        #     self.record_keep_masks(self.keep_masks)
        # handles = self.apply_prune_mask(self.model, self.keep_masks)
        # else:

        
        # init_params = self.get_model_params()
        # model.to(device)        
        

        # self.record_keep_masks(keep_masks)

        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        # self.rebuild_net(model, keep_masks)
        for h in handles:
                h.remove()
        # if self.keep_masks == None: 
        #     for h in handles:
        #         h.remove()

        # finish_params = self.get_model_params()
        # count = 0
        # for param_tensor in init_params:
        #     diff_params = init_params[param_tensor] - finish_params[param_tensor]
        #     diff_params = diff_params.view(-1).numpy()
        #     # print(diff_params)
        #     count += sum(np.where(diff_params, 0, 1))
        #     # logging.info(sum(np.where(diff_params, 0, 1)))
        # logging.info(count)

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
