function position = initialize_position(SearchAgents_no,dim,ub,lb)
% initialize position of each generation wolves
% SearchAgents_no,The number of search agents
% dim,number of your variables
% ub,Upper bound of your parameters
% lb,Lower bound of your parameters

position=zeros(SearchAgents_no,dim);
boundary_number=size(ub,2);                 % get boundary number
% if parameter just have one
if boundary_number==1
 position=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end
% if paremeter more than one
if boundary_number>1
    for i=1:dim
        position(:,i)=rand(SearchAgents_no,1).*(ub(i)-lb(i))+lb(i);
    end
end
end