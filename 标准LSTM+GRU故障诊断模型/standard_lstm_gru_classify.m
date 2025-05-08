clc
clear;close all
%% 导入数据，生成乱序数组
load F:\matalable文件\毕业设计\数据\PCA处理后数据+特征隐藏后数据\data.mat
iris_data=data_finish;         %导入数据
arry=randperm(10830);
%% 划分数据
input_train = iris_data(arry(1:7581),1:4)';
output_train = iris_data(arry(1:7581),5)';
input_test = iris_data(arry(7582:end),1:4)';
output_test = iris_data(arry(7582:end),5)';

% 节点个数
input_num = 4;                    %4个特征
hidden_num = 64;                   %隐藏层神经元
output_num = 4;                   %输出类别
%% 把输出1维转换为4维[1 0 0 0] [0 1 0 0] [0 0 1 0] [0 0 0 1]
output=zeros(7581,4);
for i=1:7581
    switch output_train(i)
        case 1
        output(i,:)=[1 0 0 0];
        case 2
        output(i,:)=[0 1 0 0];
        case 3
        output(i,:)=[0 0 1 0];
        case 4
        output(i,:)=[0 0 0 1];
    end
end
[xx,output]=max(output,[],2);

output=categorical(output);
output2=categorical(output_test);
%可以使用ind2vec函数
%% 数据归一化
method=@mapminmax;                 %最大归一化
%method=@mapstd;                   %标准归一化
[input,inputs] = method(input_train);
%[output,outputs] = method(output_train);
input_test_guiyi = method('apply',input_test,inputs);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
input =  double(reshape(input, 4, 1, 1, 7581));
input_test_guiyi  =  double(reshape(input_test_guiyi , 4, 1, 1, 3249));
Input=cell(7581,1);
Input_test_guiyi=cell(3249,1);
for i = 1 : 7581
    Input{i, 1} = input(:, 1, 1, i);
end
for i=1:3249
    Input_test_guiyi{i, 1} = input_test_guiyi( :, 1, 1, i);
end
% lstm神经网络层
numFeatures=size(input,1);
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(113,'OutputMode','last','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')       
    % LSTM层，64个隐藏单元 'sequence'输出一个序列，'last'输出最后一个时间步'sequence'输出一个序列，'last'输出最后一个时间步,对输入权重和循环权重采用HE初始化方法，有效避免梯度爆炸
    reluLayer                              %reluLayer激活函数，避免梯度爆炸
    fullyConnectedLayer(4)                 % 全连接层 对于输出结果的大小，对于分类任务后面一般需要加一个softmax层和分类层
    softmaxLayer
    classificationLayer
];
% gru神经网络层
layers1 = [  
    sequenceInputLayer(numFeatures)
    gruLayer(7,'OutputMode','last','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')       
    % LSTM层，64个隐藏单元 'sequence'输出一个序列，'last'输出最后一个时间步'sequence'输出一个序列，'last'输出最后一个时间步,对输入权重和循环权重采用HE初始化方法，有效避免梯度爆炸
    reluLayer                              %reluLayer激活函数，避免梯度爆炸
    fullyConnectedLayer(4)                 % 全连接层 对于输出结果的大小，对于分类任务后面一般需要加一个softmax层和分类层
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
 'MaxEpochs',10, ...
 'MiniBatchSize',204,...
 'GradientThreshold',1, ...               %梯度阈值用于防止梯度爆炸问题。当梯度超过这个阈值时，会被截断，通常为1-5
 'InitialLearnRate',0.0855, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',250, ...
 'LearnRateDropFactor',0.2, ...
 'L2Regularization',0.0798,...              %正则化因子
 'Verbose',false, ...
 'Plots','none'...
    );
% options = trainingOptions('adam', ...       % Adam 梯度下降算法
%     'MaxEpochs', 1000, ...                  % 最大迭代次数
%     'InitialLearnRate', 0.01, ...           % 初始学习率
%     'LearnRateSchedule', 'piecewise', ...   % 学习率下降
%     'LearnRateDropFactor', 0.1, ...         % 学习率下降因子
%     'LearnRateDropPeriod', 750, ...         % 经过 750 次训练后 学习率为 0.01 * 0.1
%     'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
%     'ValidationPatience', Inf, ...          % 关闭验证
%     'Plots', 'training-progress', ...       % 画出曲线
%     'Verbose', false);

accurancy_rate1=zeros(1,10);
accurancy_rate2=zeros(1,10);
accurancy_rate3=zeros(1,10);
accurancy_rate4=zeros(1,10);
macro_f1=zeros(1,10);
macro_f1_2=zeros(1,10);
%% lstm训练计算
for i=1:10
fprintf("LSTM迭代次数：%d\n",i);
net = trainNetwork(Input,output,layers,options);       %训练网络
t_sim1=predict(net,Input);                    %数据后处理
t_sim2=predict(net,Input_test_guiyi);
T_sim1=vec2ind(t_sim1')';
T_sim2=vec2ind(t_sim2')';
accurancy_rate1(1,i)=sum((T_sim1==output_train'))/7581*100;
accurancy_rate2(1,i)=sum((T_sim2==output_test'))/3249*100;
% end

% F1_Score
  C=confusionmat(output_test',T_sim2);                     %通过混淆矩阵取出其中的召回率和精确率
if exist("precision","var")==0
  precision = zeros(4, 10);                                %精确率/查准率
  recall = zeros(4, 10);                                   %召回率/查全率
  f1_score = zeros(4, 10);                                 %F1-Score
end

for j=1:4                                                %分别求每一类作为正的时候的召回率和精确率
    TP=C(j,j);                                           %真实例
    FP=sum(C(:,j))-TP;                                   %假正例
    FN=sum(C(j,:))-TP;                                   %假负例
    if TP+FP>0
       precision(j,i)=TP/(TP+FP);
    else
       precision(j,i)=0;
    end
    if TP+FN>0
       recall(j,i)=TP/(TP+FN);
    else
       precision(j,i)=0;
    end
%计算F1得分
    if precision(j,i)+recall(j,i)>0
       f1_score(j,i)=2*(precision(j,i)*recall(j,i))/(precision(j,i)+recall(j,i));
    else
       f1_score(j,i)=0;
    end
end
%计算宏观均值
 macro_f1(1,i)=mean(f1_score(:,i));

end
%% gru训练计算
for i=1:10
fprintf("GRU迭代次数：%d\n",i);
net2 = trainNetwork(Input,output,layers1,options);
t_sim3=predict(net2,Input);                    %数据后处理
t_sim4=predict(net2,Input_test_guiyi);
T_sim3=vec2ind(t_sim3')';
T_sim4=vec2ind(t_sim4')';
accurancy_rate3(1,i)=sum((T_sim3==output_train'))/7581*100;
accurancy_rate4(1,i)=sum((T_sim4==output_test'))/3249*100;
% end

% F1_Score
C2=confusionmat(output_test',T_sim4);                        %通过混淆矩阵取出其中的召回率和精确率
if exist("precision","var")==0                               %exist判断变量是否已经存在，存在为1，不存在为0
  precision_2 = zeros(4, 10);                                %精确率/查准率
  recall_2 = zeros(4, 10);                                   %召回率/查全率
  f1_score_2 = zeros(4, 10);                                 %F1-Score
end

for j=1:4                                                    %分别求每一类作为正的时候的召回率和精确率
    TP_2=C2(j,j);                                            %真实例
    FP_2=sum(C2(:,j))-TP_2;                                  %假正例
    FN_2=sum(C2(j,:))-TP_2;                                  %假负例
    if TP_2+FP_2>0
       precision_2(j,i)=TP_2/(TP_2+FP_2);
    else
       precision_2(j,i)=0;
    end
    if TP_2+FN_2>0
       recall_2(j,i)=TP_2/(TP_2+FN_2);
    else
       recall_2(j,i)=0;
    end
%计算F1得分
    if precision_2(j,i)+recall_2(j,i)>0
       f1_score_2(j,i)=2*(precision_2(j,i)*recall_2(j,i))/(precision_2(j,i)+recall_2(j,i));
    else
       f1_score_2(j,i)=0;
    end
end
%计算宏观均值
 macro_f1_2(1,i)=mean(f1_score_2(:,i));
end
%% 排序
[output_train,N]=sort(output_train);
TT_sim1=zeros(7581,1);
TT_sim3=zeros(7581,1);
for i=1:7581
TT_sim1(i,1)=T_sim1(N(1,i));
TT_sim3(i,1)=T_sim3(N(1,i));
end

[output_test,I]=sort(output_test);
TT_sim2=zeros(3249,1);
TT_sim4=zeros(3249,1);
for i=1:3249
TT_sim2(i,1)=T_sim2(I(1,i));
TT_sim4(i,1)=T_sim4(I(1,i));
end
%% 绘图
% LSTM绘图
figure
plot(1:7581,output_train,'r-*',1:7581,TT_sim1,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('train真实值','LSTM预测值');

figure
plot(1:3249,output_test,'r-*',1:3249,TT_sim2,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('test真实值','LSTM预测值');
% GRU绘图
figure
plot(1:7581,output_train,'r-*',1:7581,TT_sim3,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('train真实值','GRU预测值');

figure
plot(1:3249,output_test,'r-*',1:3249,TT_sim4,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('test真实值','GRU预测值');
%准确率和F1-Score对比
figure
plot(1:10,accurancy_rate2,'g',1:10,accurancy_rate4,'r');
xlabel('预测次数');
ylabel('准确率');
legend('LSTM准确率','GRU准确率');

figure
plot(1:10,macro_f1*100,'g',1:10,macro_f1_2*100,'r');
xlabel('预测次数');
ylabel('准确率');
legend('LSTM F1-Score','GRU F1-Score');

figure
subplot(2,2,1)
plot(1:10,precision(1,:),'g',1:10,precision(2,:),'r',1:10,precision(3,:),'b', ...
     1:10,precision(4,:),'k');
xlabel('预测次数');
ylabel('精确率');
title("LSTM精确率");
legend('normal data','stack leakage',"blower fault",'heat exchanger fault');

subplot(2,2,2)
plot(1:10,recall(1,:),'g',1:10,recall(2,:),'r',1:10,recall(3,:),'b', ...
     1:10,recall(4,:),'k');
xlabel('预测次数');
ylabel('召回率');
title("LSTM召回率");
legend('normal data','stack leakage',"blower fault",'heat exchanger fault');

subplot(2,2,3)
plot(1:10,precision_2(1,:),'g',1:10,precision_2(2,:),'r',1:10,precision_2(3,:),'b', ...
     1:10,precision_2(4,:),'k');
xlabel('预测次数');
ylabel('精确率');
title("GRU精确率");
legend('normal data','stack leakage',"blower fault",'heat exchanger fault');

subplot(2,2,4)
plot(1:10,recall_2(1,:),'g',1:10,recall_2(2,:),'r',1:10,recall_2(3,:),'b', ...
     1:10,recall_2(4,:),'k');
xlabel('预测次数');
ylabel('召回率');
title("GRU召回率");
legend('normal data','stack leakage',"blower fault",'heat exchanger fault');