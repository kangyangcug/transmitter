import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#参数说明
#TR  :  选出的训练点 ,    [145,145]  ,参照论文中的train,test表
#TE  :  选出的测试点 ,    [145.145]  ,参照论文中的train,test表
#input : 高光谱图片,      [145,145,200]
#label : train和test对应的标签，[145,145]
#total_pos_train: 所有训练用的pixel对应的坐标，[695,2] ,695个点，2维坐标
#total_pos_test:  所有验证用的pixel对应的坐标，[9671,2] ,695个点，2维坐标
#total_pos_true:  所有有标签的点，2维坐标,[21025,2]
#number_train, number_test, number_true 对应训练集，测试集，所有有效点的各类别样本数
#x_train ,[695,7,7,200]
#x_test ,[9671,7,7,200]
#x_true, [21025,7,7,200]
#x_train_band,[695,49*3,200]    *3是填充了一个波段点相邻两个波段的patch
#x_test_band,[9671,147,200]
#x_true_band,[21025,147,200]
#y_train,y_test,y_true对应各个样本的标签
#Label_train,Label_test,Label_true为Data.TensorDataset对象，即对应的样本和标签进行合并了
#batch_pred,[64,16],预测出的各个类别的概率
#batch_target,[64],一个batch内各个样本的标签
#train_acc在训练集上的精确度
#train_obj在训练集上的loss
#tar_t 真实值
# pre_t 预测值

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Houston', help='dataset to use')
# parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--band_patches', type=int, default=7, help='number of related band')
parser.add_argument('--epoches', type=int, default=480, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, num_classes):
    number_train = []
    pos_train = {}           #记录训练样本对应的索引位置
    number_test = []
    pos_test = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])       #统计各类别对应的样本数
        pos_train[i] = each_class   #pos_train[i]对应第i+1类

    total_pos_train = pos_train[0]          #第1类对应的位置

    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)   #np.r,按列拼接两个矩阵，要求列数相等
    total_pos_train = total_pos_train.astype(int)           #total_pos_train 标出了所有训练点的二维坐标
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)

    return total_pos_train, total_pos_test, number_train, number_test
#-------------------------------------------------------------------------------
# 边界拓展：镜像,height,width两个维度，上下左右各拓展patch//2
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #对称式填充  类似(321)0123 ，‘（321)'为参照0右边的对称进行填充的
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    #point对应total_pos_train,total_pos_test,total_pos_true中的一个
    #获取选点周围的像素
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    #该方法获得[695,49*3,200]的输出，三个49分别为该band相邻的两个band，第一个前和最后一个相邻，如第0个band和第199以及第2个band相邻
    #x_train,[695,7,7,200]
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)   #[695,49,200],另,x_train的shape不会变化
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)     #[695,49*3,200]
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):

        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    #train_point对应total_pos_train,[695,2]
    #test_point对应total_pos_test,[9671,2]
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)

    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))

    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)

    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))

    print("**************************************************")
    return x_train_band, x_test_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("**************************************************")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)      #pred,[64,1]

  pred = pred.t()     #拼接成了[1,64]

  correct = pred.eq(target.view(1, -1).expand_as(pred))    #correct为预测值与真实值比较，[1,64]的True,False矩阵

  res = []

  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)    #一个batch正确的个数
    res.append(correct_k.mul_(100.0/batch_size))       #每个batch的精确度

  return res, target, pred.squeeze()         #res为list,一个batch的精确度,target和pred.squeeze为[64],分别为真实值和预测值
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()           #梯度清0
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)    #计算loss值
        loss.backward()               #根据loss值计算反向传播梯度
        optimizer.step()             #根据梯度更新网络参数

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))         #pred1,一维list,为一个batch的精确度，t为target真实值，p为预测值,都为[64]
        n = batch_data.shape[0]    #一个batch的大小
        objs.update(loss.data, n)                 #obj统计每个batch的loss，并用于计算一个epoch的loss
        top1.update(prec1[0].data, n)                #top1统计每个batch的精度，并用于计算一个epoch的精度
        tar = np.append(tar, t.data.cpu().numpy())      #合并每个batch的真实值
        pre = np.append(pre, p.data.cpu().numpy())        #合并每个batch的预测值
    return top1.avg, objs.avg, tar, pre               #返回一个epoch的精度，loss，真实标签，预测值
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):              #flag为test的时候使用
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def show_ConfusionMatrix(tar,pre):
    #tar:target,标签
    #pre:预测值
    matrix = confusion_matrix(tar, pre)
    disp = ConfusionMatrixDisplay(matrix)
    disp.plot(include_values=True,
              cmap="viridis",
              ax=None,
              xticks_rotation="horizontal",
              values_format="d")
    plt.show()
#-------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)    #指定cuda设备的随机种子
cudnn.deterministic = True      #运行前根据输入找最优算法进行加速
cudnn.benchmark = False
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')
# print("#############################################")
# print(data)
# print("#############################################")

TR = data['TR']              #train data
TE = data['TE']              #test data
input = data['input'] #(145,145,200)
np.set_printoptions(threshold=np.inf) #将数组的元素全部打印出来
label = TR + TE
# print(label)
num_classes = np.max(TR)
print("num_classes:"+str(num_classes))
# color_mat_list = list(color_mat)
# color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
print("TR.shape:"+str(TR.shape))
print("TE.shape:"+str(TE.shape))
print("Data.shape:"+str(input.shape))
print("label_shape:"+str(label.shape))

#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band  = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
#-------------------------------------------------------------------------------
# load data
x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train=Data.TensorDataset(x_train,y_train)

x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)


#-------------------------------------------------------------------------------
# create model
model = ViT(
    image_size = args.patches,
    near_band = args.band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 64,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = args.mode
)
#最大全局精度
bestOA = 0
#获得最大精度的epoch_index
bestEpoch = 0
model = model.to(device)
# criterion
criterion = nn.CrossEntropyLoss().to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
#-------------------------------------------------------------------------------


print("start training")
tic = time.time()
for epoch in range(args.epoches):
    # scheduler.step()

    # train model
    model.train()           #train的时候会dropout,并使得normal层对每个batch独立计算方差均值
    train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)   #精度,损失值,真实标签,预测值

    OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)     #利用预测标签与真实标签计算得到混淆矩阵
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                    .format(epoch+1, train_obj, train_acc))

    scheduler.step()
    if ((epoch+1) % args.test_freq == 0) | (epoch == args.epoches - 1):
        model.eval()      #固定住BN的均值，方差，以及禁掉dropout
        tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        if(OA2>bestOA):
            bestOA = OA2
            bestEpoch=epoch
            print("current bestOA:"+str(bestOA))


        print("#############################")
        print("Current result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print(AA2)
        print("#############################")

        # if(epoch>=args.epoches-1):
        #     show_ConfusionMatrix(tar_v,pre_v)

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("**************************************************")

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print("bestOA in evaluate set:"+str(bestOA))

print(AA2)
print("**************************************************")
# print("Parameter:")
#
# def print_args(args):
#     for k, v in zip(args.keys(), args.values()):
#         print("{0}: {1}".format(k,v))
#
# print_args(vars(args))









