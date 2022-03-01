import copy
import logging
import random
import time

import numpy as np
import torch
import wandb
import json

from .utils import transform_list_to_tensor


class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        
        self.last_keep_masks = None

        self.keep_masks_dict = dict()

        self.pruned = False
        
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, local_keep_masks):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True
        self.keep_masks_dict[index] = local_keep_masks

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def merge_local_masks(self, keep_masks_dict):
        for i in range(1, len(self.keep_masks_dict[0])):
            for j in range(len(self.keep_masks_dict.keys())):
                # params = self.keep_masks_dict[0][j].to('cpu')
                self.keep_masks_dict[0][i] += self.keep_masks_dict[j][i].to('cuda:0')
                # if j == len(self.keep_masks_dict.keys())-1:
                #     diff = self.keep_masks_dict[0][i].view(-1).cpu().numpy()
                #     self.keep_masks_dict[0][i] = np.where(diff, 0, 1)
        return self.keep_masks_dict[0]

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0
        last_params = self.get_global_model_params()
        # torch.save(last_params, './debug/last_model.pt')

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        #     avg_p = averaged_params[k].view(-1).numpy()
        #     count += sum(np.where(avg_p, 0, 1))
        # logging.info(f'*********count: {count}')
        # update the global model which is cached at the server side
        # self.set_global_model_params(averaged_params)

        # count = 0
        # for param_tensor in averaged_params:
        #     # diff_params = init_model.state_dict()[param_tensor] - comp_model.state_dict()[param_tensor]
        #     diff_params = averaged_params[param_tensor].view(-1).numpy()
        #     # print(diff_params)
        #     count += sum(np.where(diff_params, 0, 1))
        #     # break
        # # print(count)
        # logging.info(f'*********count: {count}')

        # logging.info(averaged_params)

        # before_params = copy.deepcopy(last_params)
        # before_params = last_params.copy()

        for param_tensor in last_params:
            # logging.info(last_params[param_tensor].type())
            # logging.info('*********************')
            # logging.info(averaged_params[param_tensor].type())
            # break
            # p = copy.deepcopy(last_params[param_tensor])
            # logging.info(last_params[param_tensor])
            # logging.info('++++++++++++++++++')
            # logging.info(averaged_params[param_tensor])
            assert (last_params[param_tensor].shape == averaged_params[param_tensor].shape)
            # last_params[param_tensor] = last_params[param_tensor].type(torch.FloatTensor)
            last_params[param_tensor] = last_params[param_tensor].type_as(averaged_params[param_tensor])

            # averaged_params[param_tensor] = averaged_params[param_tensor].type_as(last_params[param_tensor])

            # b = last_params[param_tensor] - p
            # logging.info(f'$$$$$$$$$$$: {sum(b)}')
            last_params[param_tensor] += averaged_params[param_tensor]
            # logging.info('=============')
            # logging.info(last_params[param_tensor])

        # count = 0
        # for param_tensor in last_params:
        #     diff = (last_params[param_tensor] - before_params[param_tensor]).view(-1).numpy()
        #     count += sum(np.where(diff, 0, 1))
        # logging.info(f'&&&&&&&&&&&&&diff: {count}')


        self.set_global_model_params(last_params)
        # if not self.pruned:
        keep_masks = self.merge_local_masks(self.keep_masks_dict)
        # zeros = 0
        # for mask in keep_masks:
        #     zeros += sum(np.where(mask.view(-1).cpu().numpy(), 0, 1))
        # logging.info('*****************************************************before')
        # logging.info(f'^^^^^^^^^^^^^^zeros:{zeros}')
        # logging.info('*****************************************************after')
        self.apply_network_masks(keep_masks)
            # self.pruned = True
        # if not self.pruned:
        #     self.apply_network_masks()
        #     self.pruned = True

        # if self.last_keep_masks != None:
        #     # for j in range(len(self.keep_masks_dict.keys())):
        #     # self.record_keep_masks(self.last_keep_masks)
        #     # self.record_keep_masks(self.keep_masks_dict[0])
        #     count = 0
        #     for i in range(len(self.keep_masks_dict[0])):
        #         diff = self.last_keep_masks[i].cpu().view(-1).numpy() - self.keep_masks_dict[0][i].cpu().view(-1).numpy()
        #         count += sum(np.where(diff, 0, 1))
        #     logging.info(f'$$$$$$$$$$$$$$$$$$$$$$$$$$$diff: {count}')

        # self.last_keep_masks = self.keep_masks_dict[0]
        # self.record_keep_masks(self.last_keep_masks)
        # logging.info(self.last_keep_masks[0].shape)
        # self.record_keep_masks(self.keep_masks_dict[0])

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return last_params

    def record_keep_masks(self, keep_masks):
        masks = torch.cat([torch.flatten(x) for x in keep_masks]).to('cpu').tolist()
        with open(f'./{0}_keep_masks.txt', 'a+') as f:
            f.write(json.dumps(masks))
            f.write('\n')

    def apply_network_masks(self, keep_masks):
        self.trainer.apply_network_masks(keep_masks)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
