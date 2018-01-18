#encoding:utf-8
import data_pro as pro
import numpy as np
import torch
import model
import torch.utils.data as D
from torch.autograd import Variable
import torch.nn.functional as F
import random

data = pro.load_data('new_train.txt') #


word_dict,maxlen=pro.build_dict(data[0])
embed_num=len(word_dict)
#pos_dict,_=pro.build_dict(data[1])
#pos_size=len(pos_dict)
#loc_dict,_=pro.build_dict(data[2])
#loc_size=len(loc_dict)
#print loc_dict.items()
label_dict=pro.build_label(data[1])
class_num=len(label_dict)
#print embed_num,pos_size,class_num


import argparse
parser=argparse.ArgumentParser(description='question classification')
parser.add_argument('-max_len',type=int,default=maxlen)
parser.add_argument('-embed_dim',type=int,default=150)
parser.add_argument('-embed_num',type=int,default=embed_num)
parser.add_argument('-dropout',type=float,default=0.4)
parser.add_argument('-hidden_size',type=int,default=100)
#parser.add_argument('-pos_size',type=int,default=pos_size)
#parser.add_argument('-pos_dim',type=int,default=100)

#parser.add_argument('-loc_size',type=int,default=loc_size)
#parser.add_argument('-loc_dim',type=int,default=100)
parser.add_argument('-batch_size',type=int,default=4)
parser.add_argument('-class_num',type=int,default=class_num)
parser.add_argument('-epochs',type=int,default=2000)
parser.add_argument('-t_size',type=int,default=100)
parser.add_argument('-train',type=str,default='true')
parser.add_argument('-f',type=str)


args=parser.parse_args()


#model1=model.BiLSTM(args).cuda()
#model1=BiLSTM(args).cuda()
model1=model.BiLSTM(args)
print model1
optimizer = torch.optim.Adam(model1.parameters())
loss_func=torch.nn.CrossEntropyLoss()

def data_unpack(cat_data, target):
    list_x = np.split(cat_data.numpy(), [args.max_len,2*args.max_len], 1)
    #np.split(x,[p1,p2,p3,p4,...],dim)表示第几列
    x = Variable(torch.from_numpy(list_x[0])).cuda()
    pos = Variable(torch.from_numpy(list_x[1])).cuda()
    loc = Variable(torch.from_numpy(list_x[2])).cuda()
    target = Variable(target).cuda()
    return x, pos,loc, target

def prediction(out, y):
    predict = torch.max(out, 1)[1].long()
    correct = torch.eq(predict, y)
    acc = correct.sum().float() / float(correct.data.size()[0])
    return (acc * 100).cpu().data.numpy()[0],predict


###############################################################
'''training data'''
#print type(data[2])
#x,pos,loc,y=pro.vectorize(data,word_dict,pos_dict,loc_dict,label_dict,args.max_len)
x,y=pro.vectorize(data,word_dict,label_dict,args.max_len)
y = np.array(y).astype(np.int64)
#np_cat = np.concatenate((x, pos,loc),1)

#train = torch.from_numpy(np_cat.astype(np.int64))
train = torch.from_numpy(x.astype(np.int64))

y_tensor = torch.LongTensor(y)

train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)

train_dataloader = D.DataLoader(train_datasets, args.batch_size, shuffle=True, num_workers=2)

###############################################################
'''test data'''
#t_data = pro.load_data('segtest.txt','testnature.txt','test_weizhi_information_encode','test_coarse_type.txt')
t_data = pro.load_data('new_test.txt')
#t_x,t_pos,t_loc,t_y=pro.vectorize(t_data,word_dict,pos_dict,loc_dict,label_dict,args.max_len)
t_x,t_y=pro.vectorize(t_data,word_dict,label_dict,args.max_len)
t_y = np.array(t_y).astype(np.int64)
#t_np_cat = np.concatenate((t_x, t_pos,t_loc),1)
#t_np_cat = np.concatenate((t_x, t_pos,t_loc),1)

#test = torch.from_numpy(t_np_cat.astype(np.int64))
test = torch.from_numpy(t_x.astype(np.int64))
t_y_tensor = torch.LongTensor(t_y)
test_datasets = D.TensorDataset(data_tensor=test, target_tensor=t_y_tensor)
test_dataloader = D.DataLoader(test_datasets, args.batch_size, shuffle=False, num_workers=2)

# index = [i for i in range(len(x))]
# random.shuffle(index)
# data = [x[ind] for ind in index]
# pos=[pos[index]for ind in index]
# label = [y[index]for ind in index]




if args.train=='true':
    output = open('test.log', 'w+')
    output.write('-' * 50 + '\n')
    output.flush()
    max_acc = 0.0
    step = 0
    for i in range(args.epochs):
       # print "epochs:",i
        acc=0.0
        l=0.0
        k=0
        for (x_cat, y) in train_dataloader:
            #x,pos,loc,y = data_unpack(x_cat, y)
            #print 'x:', x, 'pos:', pos
            #print y
            #out = model1(x, pos,loc)
			print x_cat
			x_cat=Variable(x_cat)
			y=Variable(y)
			out=model1(x_cat)
            #print out.data,y
			loss = loss_func(out, y)
			l += loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print loss,prediction(out,y)
			pre,_=prediction(out,y)
			acc+=pre
			k+=1
        print 'epoch:', i, 'acc:', acc / k, '%   loss:', l.cpu().data.numpy()[0] / k
        test_acc=0.0
        j=0
        test_l=0.0
        result=[]
        y_test=[]
        for (t_x_cat,t_y) in test_dataloader:
            #t_x,t_pos,t_loc,t_y=data_unpack(t_x_cat,t_y)
            #t_out=model1(t_x,t_pos,t_loc)
			t_x_cat=Variable(t_x_cat)
			t_y=Variable(t_y)
			t_out=model1(t_x_cat)
			loss = loss_func(t_out, t_y)
			test_l += loss
			t_pre,predic=prediction(t_out,t_y)
            #print 'predic:',predic.data,'t_y:',t_y.data
			test_acc+=t_pre
			j+=1
			result+=list(predic.cpu().data.numpy())
			y_test+=list(t_y.cpu().data.numpy())
        print 'epoch:', i, 'test_acc:', test_acc / j, '%   loss:', test_l.cpu().data.numpy()[0] / j
        #print label_dict.items()
        new_label_dict={}
        for key,value in label_dict.items():
            new_label_dict[value]=key
        output.write('epoch:'+str(i)+'test_acc:'+str(test_acc / j)+ '%   loss:'+str(test_l.cpu().data.numpy()[0] / j)+'\n')
        output.flush()

        if test_acc/j>max_acc:
            step=i
            max_acc=test_acc/j
            torch.save(model1.state_dict(),'model.pt')
            with open('result.txt','w') as f_test:
                print 'max_acc:---------------------------------------------------------------',max_acc


                for ind in result:
                    #print ind,new_label_dict[ind],y_test[i],t_data[2][i]
                    #f_test.write((label_dict.keys()[ind]+'\n').encode('utf-8'))
                    f_test.write((new_label_dict[ind]+'\n').encode('utf-8'))
        if i>=args.epochs-1:
            output.write('max_acc:'+'-'*40+str(max_acc)+'\n')
            output.flush()
            output.close()
if args.train=='false':
    print 'testing......'
    model1.load_state_dict(torch.load('model.pt'))
    test_acc = 0.0
    j = 0
    test_l = 0.0
    result = []
    y_test = []
   # f_v=open('vector.csv','w')
    for (t_x_cat, t_y) in test_dataloader:
    #for (t_x_cat, t_y) in train_dataloader:
        #t_x, t_pos,t_loc, t_y = data_unpack(t_x_cat, t_y)
        #t_out = model1(t_x, t_pos,t_loc)
		
        t_out = model1(t_x_cat)
        loss = loss_func(t_out, t_y)
        test_l += loss
        t_pre, predic = prediction(t_out, t_y)
        # print 'predic:',predic.data,'t_y:',t_y.data
        test_acc += t_pre
        j += 1
        result += list(predic.cpu().data.numpy())
        y_test += list(t_y.cpu().data.numpy())
        #print len(list(vec.cpu.data.numpy()))
        #np.savetxt("v.txt",vec.data.cpu().numpy())
        #for v_i in list(vec.cpu().data.numpy()):
            #print v_i
            #for v_j in v_i:
                #print len(v_j)
               #f_v.write(('\t'.join(v_j)).encode('utf-8'))
            #f_v.write(('\n').encode('utf-8'))
    #f_v.close()
    #print 'test_acc:', test_acc / j, '%   loss:', test_l.cpu().data.numpy()[0] / j
    with open('test_result.txt','w') as f_w:
        new_label_dict = {}
        for key, value in label_dict.items():
            new_label_dict[value] = key
        for ind in result:
            # print ind,new_label_dict[ind],y_test[i],t_data[2][i]
            # f_test.write((label_dict.keys()[ind]+'\n').encode('utf-8'))
            f_w.write((new_label_dict[ind] + '\n').encode('utf-8'))