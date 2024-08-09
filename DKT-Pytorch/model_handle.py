import torch
import torch.optim as optim

from models.DKT import DKT
from loss import lossFunc

import tqdm
import time
import os

from constants import *


class ModelHandler():

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.model = None
        self.loss_fn = None
        self.optimizer = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir('outputs'):
            os.mkdir('outputs')

        print('-' * 30)
        print('-' * 10, self.device, '-' * 10)
        print('-' * 30)

        pass

    def load_model(self, model_path=None):
        if model_path == None:
            self.model = DKT(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        else:
            self.model = DKT(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
            self.model.load_state_dict(torch.load(model_path))
        pass

    def compile_model(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = lossFunc().to(self.device)
        self.model.to(self.device)
        
        print('compile model', '-' * 20)
        print(self.optimizer)
        print('')
        print(self.loss_fn)
        print('')
        print(self.model)
        print('-' * 30)

    def save_model(self, epoch, val_loss, val_acc):

        save_path = f'outputs/{epoch}_{val_loss:.4f}_{val_acc:.4f}.pth'
        torch.save(self.model.state_dict(), save_path)

        pass

    def train(self, train_generator, val_generator, n_epoch):

        min_loss = 10e+8

        for epoch in tqdm.tqdm(range(n_epoch), desc='Training:', mininterval=2):

            # training step
            running_loss, running_acc = self._optimize(train_generator, epoch, train=True)
            print(f"epoch : {epoch}/{n_epoch}  running_acc : {running_acc:.4f}, running_loss : {running_loss.item():.4f}")

            # validation step
            if epoch % 5 == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_loss, val_acc = self._optimize(val_generator, epoch, train=False)
                    print(f"epoch : {epoch}/{n_epoch}  val_acc : {val_acc:.4f}, val_loss : {val_loss.item():.4f}")

                if val_loss < min_loss:
                    min_loss = val_loss
                    self.save_model(epoch, min_loss, val_acc)
        pass

    def _optimize(self, data_generator, epoch, train=True):

        start = time.time()

        if train:

            running_loss = 0
            running_acc = 0

            self.model.train()

            for num, batch in enumerate(data_generator):

                batch = batch.to(self.device)

                # wipe any existing gradients from previous iterations
                self.optimizer.zero_grad()

                pred = self.model(batch)
                loss, acc = self.loss_fn(pred, batch)

                # this step computes all gradients with "autograd"
                # i.e. automatic differentiation
                loss.backward()

                # this actually changes the parameters
                self.optimizer.step()

                # if the current loss is better than any ones we've seen
                # before, save the parameters.

                running_loss += loss
                running_acc += acc

                end = time.time()

                if (num + 1) % 16 == 0:
                    print(
                        f"[{epoch} epoch {num + 1}/{len(data_generator)} iter] batch_running_acc : {acc:.4f}, batch_running_loss : {loss.item():.4f} time : {end - start:.2f} sec",
                        end='\r', flush=True)

            running_loss = running_loss / len(data_generator)
            running_acc = running_acc / len(data_generator)

            return running_loss, running_acc

        else:

            val_loss = 0
            val_acc = 0

            self.model.eval()

            for num, batch in enumerate(data_generator):

                batch = batch.to(self.device)

                with torch.no_grad():

                    pred = self.model(batch)
                    loss, acc = self.loss_fn(pred, batch)
                    val_loss += loss
                    val_acc += acc

                    end = time.time()

                if num % 16 == 1:
                    print(
                        f"[{epoch + 1} epoch {num + 1}/{len(data_generator)} iter] batch_val_acc : {acc:.4f}, batch_val_loss : {loss.item():.4f} time : {end - start:.2f} sec",
                        end='\r', flush=True)

            val_loss = val_loss / len(data_generator)
            val_acc = val_acc / len(data_generator)

            return val_loss, val_acc

    def evaluate(self, test_generator):
        #只跑一次
        test_loss, test_acc = self._optimize(test_generator, 0, train=False)#epoch=0
        print('-'*50)
        print(f" test_acc : {test_acc:.4f}, test_loss : {test_loss.item():.4f}")
        pass

    def predict(self, x):#输入的是某一个batch中的一个学生的数据，x=[seq_len, numofq*2]

        def _cal_prob(x):#x=[1, seq_len, numofq*2]

            # qt
            delta = x[:,:,:QUESTION_NUM] + x[:,:,QUESTION_NUM:]

            # qt+1
            delta = delta[:,1:,:].permute(0,2,1)

            # yt
            pred = self.model(x)
            y = pred[:, :MAX_SEQ - 1,:]

            # pred at+1
            temp = torch.matmul(y, delta) # 1, MAX_SEQ, MAX_SEQ-1(prob)

            # get excercise prob from diagonal matrix
            prob = torch.diagonal(temp, dim1=1, dim2=2) # 1, MAX_SEQ-1(prob)\

            return prob.squeeze(0)#再把batch维度去掉

        #这里目前有单看不太懂
        def _get_q_sequence(q_seq_one_hot):#q_seq_one_hot=[1, seq_len, numofq*2]

            q_sequence = []
            one_hot_excercise_tags = q_seq_one_hot[:, :, :QUESTION_NUM] + q_seq_one_hot[:, :, QUESTION_NUM:]#q_seq_one_hot=[1, seq_len, numofq]做题标记
            one_hot_excercise_tags = one_hot_excercise_tags.squeeze(0)#去掉batch维度

            for one_hot_excercise_tag in one_hot_excercise_tags:
                try:
                    excercise_tag = torch.nonzero(one_hot_excercise_tag).item()
                except:
                    excercise_tag = -1

                q_sequence.append(excercise_tag)

            return torch.Tensor(q_sequence)

        def _get_a_sequence(q_seq_one_hot):
            q_seq_one_hot = q_seq_one_hot.squeeze(0)
            a_sequence = ((q_seq_one_hot[:, :QUESTION_NUM] - q_seq_one_hot[:, QUESTION_NUM:]).sum(1) + 1) // 2
            return a_sequence


        if len(x.size()) == 2:#先给x增加一个维度，因为要送入模型·
            x = x.unsqueeze(0)

        x = x.to(self.device)

        prob = _cal_prob(x)

        q_sequence = _get_q_sequence(x)
        a_sequence = _get_a_sequence(x)

        print("q_sequence",q_sequence)
        print("a_sequence",a_sequence)
        '''
q_sequence tensor([146.,  87., 206.,  87.,  87., 654., 193.,  17.,  78.,  66., 158., 248.,
         17.,  78.,  66., 158., 248.,  17.,  78.,  66., 158., 248.,  17.,  78.,
         66., 158., 248.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,
         -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,
         -1.,  -1.])
a_sequence tensor([1., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       device='cuda:0')

       '''
        #last_excercise_tag 最后一个用户有效的题目标签
        if -1 in q_sequence:#如果有-1
            last_excercise_tag = torch.nonzero(q_sequence == -1)[0][0].item() - 1 #torch.nonzero(q_sequence == -1)[0][0]拿到的是第一个-1出现的位置，需要-1 是用户最后一个作答的位置 
        else:#如果没有-1，那就是用户作答了maxstep =50个问题
            last_excercise_tag = len(q_sequence) - 1#我们的索引要-1 ，50-1=49 用户最后一个作答的位置是49


        print('-' * 50)
        #用户当前作答标签序列
        print(f'sol excercise tags: \n {q_sequence[:last_excercise_tag-1]}')#为什么又-1？？ 减一操作是为了排除序列中最后一个有效的题目标签，只关注在此之前的题目。
        print(f'result excercise tags: \n {a_sequence[:last_excercise_tag-1]}')
        print('-' * 50)

        #需要预测的问题id
        ###这里暂时有点看不太懂？20240809暂时放一下吧！！！！
        print(f'predict excercise tag {q_sequence[last_excercise_tag]}')#预测的题目id 这两行代码分别打印最后一个有效题目标签（last_excercise_tag 指向的是序列中最后一个非 -1 的位置）和对应的答案结果（来自 a_sequence
        print(f'ground truth : {a_sequence[last_excercise_tag]}')#真实的题目作答结果
        print(f'this student has a {prob[last_excercise_tag-1]*100:.2f}% chance of solving this problem')#预测的作答概率 ，基于前一个题目的作答情况（即 last_excercise_tag-1），预测学生解决当前题目（last_excercise_tag 位置上的题目）的成功概率。这里使用 last_excercise_tag-1 是因为通常预测模型会利用之前的题目来预测下一个题目的作答情况
        '''
        为什么使用 last_excercise_tag-1
            根据上面的解释，prob[last_excercise_tag] 将会使用包括在 last_excercise_tag 这个位置的题目的信息来预测之后的题目，而在这段代码的上下文中，last_excercise_tag 实际上是序列中最后一个有效的题目标签。因此：

            使用 prob[last_excercise_tag-1] 是为了获取对于最后一个题目（last_excercise_tag 对应的题目）的解决概率，这个概率是基于之前所有题目的答题表现（直到 last_excercise_tag-1）计算的。
            如果使用 prob[last_excercise_tag]，它可能会预测之后一个不存在的题目，或者其计算可能不符合当前的逻辑需求，因为这通常会是超出了当前数据范围的预测。
                    
        '''

