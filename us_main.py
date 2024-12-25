import os
import yaml

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# record
from tqdm import tqdm
from tensorboardX import SummaryWriter

from util.parser import get_parser

from util.util import import_class, us_accuracy, AverageMeter

class Processor():

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()
        #self.load_state()
        self.best_train_acc1 = float('inf')
        self.best_test_acc1 = float('inf')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if self.arg.phase == 'train':
            self.trainloader = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            pin_memory=True)

    def load_model(self):
        self.device = self.arg.device
        Model = import_class(self.arg.model)
        self.model = Model(self.arg.model_args['channels']).to(self.device)
        self.loss = nn.MSELoss().to(self.device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.arg.base_lr,momentum=0.9,nesterov=self.arg.nesterov,weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.arg.base_lr,weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def load_scheduler(self):
        self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer,lr_lambda = lambda epoch: 0.95)

    def load_state(self):
        path = os.path.join(self.arg.work_dir,'checkpoint.pth')
        
        if self.arg.weights:
            ckpt = torch.load(self.arg.weights, map_location='cpu', weights_only=True)
            self.model.load_state_dict(ckpt['model'], strict=False)
        # automatically resume from checkpoint if it exists
        elif os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(ckpt['model'], strict=False)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.arg.start_epoch = ckpt['epoch']
        else:
            self.arg.start_epoch = 0

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def train(self):
        train_writer = SummaryWriter(self.arg.work_dir)
        self.model.train()
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            print("Epoch:[{}/{}]".format(epoch + 1, self.arg.num_epoch))
            loss_epoch = 0
            mse_meter = AverageMeter('MSE_train')  # 新增，用于记录均方误差平均值

            for batch_idx, data in enumerate(self.trainloader):
                data = data.float().to(self.device, non_blocking=True)

                # forward
                output = self.model.forward(data)
                loss = self.loss(output, data)

                # 计算当前批次的均方误差，并更新到mse_meter中
                mse = torch.mean((output - data) ** 2).item()
                mse_meter.update(mse, data.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n_batch = len(self.trainloader)
                loss_epoch += loss.item()

                step = epoch * n_batch + batch_idx
                train_writer.add_scalar('loss_step', loss.data.item(), step)
                train_writer.add_scalar('lr', self.scheduler.get_last_lr(), step)

            self.scheduler.step()

            print("Train MSE: {} Mean loss: {} LR: {}".format(mse_meter.avg, loss_epoch / n_batch,
                                                            self.scheduler.get_last_lr()))

            if mse_meter.avg < self.best_train_acc1:
                self.best_train_acc1 = mse_meter.avg

            self.eval()

            state = dict(epoch=epoch + 1, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict())
            torch.save(state, os.path.join(self.arg.work_dir, 'checkpoint.pth'))

            if epoch % 10 == 0:
                model_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir, epoch)
                torch.save(self.model.state_dict(), model_path)

        print("Best Train MSE: {}".format(self.best_train_acc1))


#修改点
    def eval(self, save_score=False):
        step_val = 0
        val_writer = SummaryWriter(self.arg.work_dir)
        self.model.eval()
        mse_meter = AverageMeter('MSE_val')  # 新增，用于记录验证集上的均方误差平均值
        with torch.no_grad():
            for data in self.testloader:
                data = data.float().to(self.device, non_blocking=True)

                output = self.model(data)
                loss = self.loss(output, data)

                # 计算当前批次的均方误差，并更新到mse_meter中
                mse = torch.mean((output - data) ** 2).item()
                mse_meter.update(mse, data.size(0))

                val_writer.add_scalar('loss_val', loss.data.item(), step_val)

        print("Eval MSE: {}".format(mse_meter.avg))

        if mse_meter.avg < self.best_test_acc1:
            self.best_test_acc1 = mse_meter.avg

        print("Eval Best MSE: {}".format(self.best_test_acc1))

        if save_score and self.arg.phase == 'test':
            # 根据需求保存重构输出等相关数据（这里示例保存输出数据，可根据实际调整）
            np.save('{}/reconstructed_data.npy'.format(self.arg.work_dir), output.data.cpu().numpy())


    def start(self):
        if self.arg.phase == 'train':
            print('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.train()

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.eval(save_score=self.arg.save_score)

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f,Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()
