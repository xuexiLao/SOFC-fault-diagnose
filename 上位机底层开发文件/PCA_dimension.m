function [label] = PCA_dimension(net,lambda,data,train_data,standard_data)
% PCA数据降维，并进行预测结果输出
% net:输入网络
% lambda:特征向量的输入
% data:输入数据
% train_data:用来给定网络数据标准化(训练数据）
% standard_data:标准化数据，用来PCA标准化（训练数据）

F=zeros(size(data,1),4);                         % 保留4个主成分
[n,~]=size(data);                                % 获取样本个数和样本特征数

N=standard_data(:,1:11);                         % 给定200个值作为计算标准值的标准
N=[N;data];                                      % 合并标准化数据及读取的数据
X=zscore(N);                                     % 对数据标准化
X=X(201:(200+n),:);                              % 取出读取的数据

for j=1:4
  ai=lambda(:,j)';                               % 把第i给特征向量取出，转置为行向量
  Bi=repmat(ai,n,1);                             % 把ai复制n次形成n*p的矩阵
  F(:,j)=sum(Bi.*X,2);                           % 对标准化的数据求了权重后计算每一行的和
end
% 合并最终数据
N=[rand(3,n)*5;rand(1,n)*1.0]';
F=F+N;

input_train = train_data(1:7580,1:4)';            %输入数据取出
method=@mapminmax;                                %最大归一化
[~,inputs] = method(input_train);
%[output,outputs] = method(output_train);
input_verify = method('apply',F',inputs);

m=size(F,1);
input_verify = double(reshape(input_verify , 4, 1, 1, m));
Input_verify=cell(m,1);
for k=1:m
    Input_verify{m, 1} = input_verify( :, 1, 1, k);
end
% disp(Input_verify)

t_sim1=predict(net,Input_verify);                    %数据后处理
[~,label]=max(t_sim1);

end
