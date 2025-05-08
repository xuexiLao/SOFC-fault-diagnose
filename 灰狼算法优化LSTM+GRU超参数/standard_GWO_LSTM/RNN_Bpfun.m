function err = RNN_Bpfun(x, P_train, T_train, P_test, T_test,SearchAgents_no,Max_iteration)
%% 训练与测试LSTM网络
%% 输入
% x:一个个体的初始权值和阈值
% P_train:训练样本输入
% T_train:训练样本输出
% hiddennum:隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
% SearchAgents_no:狼群数量
% Max_iteration：迭代次数
%% 获取参数
best_lr=x(1);
best_l2=x(2);
best_mb=round(x(3));                        %输入最小规模为整数，对输入参数进行四舍五入
best_hidden=round(x(4));
%% 建立LSTM网络
numFeatures=4;                             %输入特征
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(best_hidden,'OutputMode','last','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')       
    % LSTM层，64个隐藏单元 'sequence'输出一个序列，'last'输出最后一个时间步'sequence'输出一个序列，'last'输出最后一个时间步,对输入权重和循环权重采用HE初始化方法，有效避免梯度爆炸
    reluLayer                              %reluLayer激活函数，避免梯度爆炸
    fullyConnectedLayer(4)                 % 全连接层 对于输出结果的大小，对于分类任务后面一般需要加一个softmax层和分类层
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
 'MaxEpochs',20, ...
 'MiniBatchSize',best_mb,...
 'GradientThreshold',1, ...               %梯度阈值用于防止梯度爆炸问题。当梯度超过这个阈值时，会被截断，通常为1-5
 'InitialLearnRate',best_lr, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',10, ...
 'LearnRateDropFactor',0.2, ...
 'L2Regularization',best_l2 ,...
 'Verbose',false, ...
 'Plots','none'...
    );

net = trainNetwork(P_train,T_train,layers,options);             %训练网络

%% 模型评估
t_sim=predict(net,P_test);
T_sim=vec2ind(t_sim')';
C=confusionmat(T_test',T_sim);                             %通过混淆矩阵取出其中的召回率和精确率
persistent precision recall f1_score macro_f1 n accurate   %持久变量，只执行一次
if isempty(recall)                                         %判断数据是否为空（未赋值）,为空则是ture,否则为false
  precision = zeros(4, SearchAgents_no*Max_iteration);                                %精确率/查准率
  recall = zeros(4, SearchAgents_no*Max_iteration);                                   %召回率/查全率
  f1_score = zeros(4, SearchAgents_no*Max_iteration);                                 %F1-Score
  macro_f1=zeros(1,SearchAgents_no*Max_iteration);
  accurate=zeros(1,SearchAgents_no*Max_iteration);                                    %准确率
end

if isempty(n)                                           %设置索引，记录每一代的精确率和召回率
  n=0;
end
n=n+1;

for j=1:4                                                %分别求每一类作为正的时候的召回率和精确率
    TP=C(j,j);                                           %真实例
    FP=sum(C(:,j))-TP;                                   %假正例
    FN=sum(C(j,:))-TP;                                   %假负例
    if TP+FP>0
       precision(j,n)=TP/(TP+FP);
    else
       precision(j,n)=0;
    end
    if TP+FN>0
       recall(j,n)=TP/(TP+FN);
    else
       precision(j,n)=0;
    end
%计算F1得分
    if precision(j,n)+recall(j,n)>0
       f1_score(j,n)=2*(precision(j,n)*recall(j,n))/(precision(j,n)+recall(j,n));
    else
       f1_score(j,n)=0;
    end
end
macro_f1(1,n)=mean(f1_score(:,n));                                       %宏观F1得分均值
accurate(1,n)=sum(T_sim==T_test')/size(T_test,2);                        %准确率

err=-macro_f1(1,n);                                                     %适应度值，根据更小的适应度能够获得更大的影响，因此采用1-F1得分

assignin("base",'n',n);                                                  %输出到工作空间
 if n==SearchAgents_no*Max_iteration
   assignin("base",'precision',precision);
   assignin("base",'recall',recall);
   assignin("base",'macro_f1',macro_f1);%所有的F1得分
   assignin("base",'accurate',accurate);%准确率
 end

end