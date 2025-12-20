price=100:100:400;
%price=100:100:800;
number=0.2:0.2:1;
%number=0.1:0.1:1;
T=50000;%time horrizon
probability=[];
utility=[];
price_1=[];%This new price is used for MTS MUTSC MTSC algorithm which is to calculate price_1.*theta
for i=1:length(price)
    for j=1:length(number)
        probability=[probability exp(-(price(i)/100)-number(j))];
        utility=[utility price(i)*exp(-(price(i)/100)-number(j))];
        price_1=[price_1 price(i)];
    end
end
utility_norm=utility/max(price);
utility_max=max(utility);
gamma=2;%number of neighbors
n=length(utility);%number of arm
n_1=length(price);%number of cluster
n_2=length(number);%number of arms in each cluster



%TS algorithm
regret_avg_TS=[];
a_avg_TS=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
a_TS=[];%record the action for TS
r_TS=[];
for i=1:T
    theta = betarnd(1+S,1+F);
    [~,index]=max(theta);
    a_TS=[a_TS index];
    r_TS=[r_TS utility_max-utility(index)];
    X = binornd(1,utility_norm(index));
    if X==1
        S(index)=S(index)+1;
    else
        F(index)=F(index)+1;
    end
end
regret_TS=[];%caculate the cumulative regret
for i=1:T
    regret_TS=[regret_TS sum(r_TS(1:i))];
end
regret_avg_TS=[regret_avg_TS;regret_TS];
a_avg_TS=[a_avg_TS;a_TS];
end


%MTS algorithm
regret_avg_MTS=[];
a_avg_MTS=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
a_MTS=[];%record the action for TS
r_MTS=[];
for i=1:T
    theta = betarnd(1+S,1+F);
    [~,index]=max(price_1.*theta);
    a_MTS=[a_MTS index];
    r_MTS=[r_MTS utility_max-utility(index)];
    X = binornd(1,probability(index));
    if X==1
        S(index)=S(index)+1;
    else
        F(index)=F(index)+1;
    end
end
regret_MTS=[];%caculate the cumulative regret
for i=1:T
    regret_MTS=[regret_MTS sum(r_MTS(1:i))];
end
regret_avg_MTS=[regret_avg_MTS;regret_MTS];
a_avg_MTS=[a_avg_MTS;a_MTS];
end

%TSC algorithm
regret_avg_TSC=[];
a_avg_TSC=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
S_c=zeros(1,n_1);
F_c=zeros(1,n_1);
a_TSC=[];%record the action for TS
r_TSC=[];
for i=1:T
    theta_c = betarnd(1+S_c,1+F_c);
    [~,index_c]=max(theta_c);%find cluster
    theta = betarnd(1+S(((index_c-1)*n_2+1):(index_c*n_2)),1+F(((index_c-1)*n_2+1):(index_c*n_2)));
    [~,index]=max(theta);
    action=(index_c-1)*n_2+index;
    a_TSC=[a_TSC action];
    r_TSC=[r_TSC utility_max-utility(action)];
    X = binornd(1,utility_norm(action));
    if X==1
        S(action)=S(action)+1;
        S_c(index_c)=S_c(index_c)+1;
    else
        F(action)=F(action)+1;
        F_c(index_c)=F_c(index_c)+1;
    end
end
regret_TSC=[];%caculate the cumulative regret
for i=1:T
    regret_TSC=[regret_TSC sum(r_TSC(1:i))];
end
regret_avg_TSC=[regret_avg_TSC;regret_TSC];
a_avg_TSC=[a_avg_TSC;a_TSC];
end

%MTSC algorithm
regret_avg_MTSC=[];
a_avg_MTSC=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
S_c=zeros(1,n_1);
F_c=zeros(1,n_1);
l=zeros(1,n);% number of times becomes to the leader
gamma_1=gamma+1;%number of the neighbor of  the leader
a_MTSC=[];%record the action for TS
r_MTSC=[];
for i=1:n
    X = binornd(1,utility_norm(i));
    X_1 = binornd(1,probability(i));
    a_MTSC=[a_MTSC i];
    r_MTSC=[r_MTSC utility_max-utility(i)];
    if X==1
        S_c(ceil(i/n_2))=S_c(ceil(i/n_2))+1;
    else
        F_c(ceil(i/n_2))=F_c(ceil(i/n_2))+1;
    end
    if X_1==1
        S(i)=S(i)+1;
    else
        F(i)=F(i)+1;
    end
end
for i=(n+1):T
    theta_c = betarnd(1+S_c,1+F_c);
    [~,index_c]=max(theta_c);%find cluster
    mu=S(((index_c-1)*n_2+1):(index_c*n_2))./(S(((index_c-1)*n_2+1):(index_c*n_2))+F(((index_c-1)*n_2+1):(index_c*n_2)));
    [~,leader_1]=max(mu);
    leader=(index_c-1)*n_2+leader_1;
    l(leader)=l(leader)+1;
    if mod(l(leader),gamma_1)==0
        a_MTSC=[a_MTSC leader];
        r_MTSC=[r_MTSC utility_max-utility(leader)];
        X = binornd(1,utility_norm(leader));
        X_1 = binornd(1,probability(leader));
        if X==1
           S_c(index_c)=S_c(index_c)+1;
        else
           F_c(index_c)=F_c(index_c)+1;
        end
        if X_1==1
           S(leader)=S(leader)+1;
        else
           F(leader)=F(leader)+1;
        end
    else
        if leader_1==1
           theta = betarnd(1+S(leader:leader+1),1+F(leader:leader+1));
           [~,index]=max(theta);
           a_MTSC=[a_MTSC leader-1+index];
           r_MTSC=[r_MTSC utility_max-utility(leader-1+index)];
           X = binornd(1,utility_norm(leader-1+index));
           X_1 = binornd(1,probability(leader-1+index));
           if X==1
              S_c(index_c)=S_c(index_c)+1;
           else
              F_c(index_c)=F_c(index_c)+1;
           end
           if X_1==1
              S(leader-1+index)=S(leader-1+index)+1;
           else
              F(leader-1+index)=F(leader-1+index)+1;
           end
        elseif leader_1==n_2
           theta = betarnd(1+S((leader-1):leader),1+F((leader-1):leader));
           [~,index]=max(theta);
           a_MTSC=[a_MTSC leader+index-2];
           r_MTSC=[r_MTSC utility_max-utility(leader+index-2)];
           X = binornd(1,utility_norm(leader+index-2));
           X_1 = binornd(1,probability(leader+index-2));
           if X==1
              S_c(index_c)=S_c(index_c)+1;
           else
              F_c(index_c)=F_c(index_c)+1;
           end
           if X_1==1
              S(leader+index-2)=S(leader+index-2)+1;
           else
              F(leader+index-2)=F(leader+index-2)+1;
           end
        else
           theta = betarnd(1+S(leader-1:leader+1),1+F(leader-1:leader+1));
           [~,index]=max(theta);
           a_MTSC=[a_MTSC leader+index-2];
           r_MTSC=[r_MTSC utility_max-utility(leader+index-2)];
           X = binornd(1,utility_norm(leader+index-2));
           X_1 = binornd(1,probability(leader+index-2));
           if X==1
              S_c(index_c)=S_c(index_c)+1;
           else
              F_c(index_c)=F_c(index_c)+1;
           end
           if X_1==1
              S(leader+index-2)=S(leader+index-2)+1;
           else
              F(leader+index-2)=F(leader+index-2)+1;
           end
        end
    end
end
regret_MTSC=[];%caculate the cumulative regret
for i=1:T
    regret_MTSC=[regret_MTSC sum(r_MTSC(1:i))];
end
regret_avg_MTSC=[regret_avg_MTSC;regret_MTSC];
a_avg_MTSC=[a_avg_MTSC;a_MTSC];
end

%UTSC algorithm
regret_avg_UTSC=[];
a_avg_UTSC=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
S_c=zeros(1,n_1);
F_c=zeros(1,n_1);
l=zeros(1,n);% number of times becomes to the leader
gamma_1=gamma+1;%number of the neighbor of  the leader
a_UTSC=[];%record the action for TS
r_UTSC=[];
for i=1:n
    X = binornd(1,utility_norm(i));
    a_UTSC=[a_UTSC i];
    r_UTSC=[r_UTSC utility_max-utility(i)];
    if X==1
        S(i)=S(i)+1;
        S_c(ceil(i/n_2))=S_c(ceil(i/n_2))+1;
    else
        F(i)=F(i)+1;
        F_c(ceil(i/n_2))=F_c(ceil(i/n_2))+1;
    end
end
for i=(n+1):T
    theta_c = betarnd(1+S_c,1+F_c);
    [~,index_c]=max(theta_c);%find cluster
    mu=S(((index_c-1)*n_2+1):(index_c*n_2))./(S(((index_c-1)*n_2+1):(index_c*n_2))+F(((index_c-1)*n_2+1):(index_c*n_2)));
    [~,leader_1]=max(mu);
    leader=(index_c-1)*n_2+leader_1;
    l(leader)=l(leader)+1;
    if mod(l(leader),gamma_1)==0
        a_UTSC=[a_UTSC leader];
        r_UTSC=[r_UTSC utility_max-utility(leader)];
        X = binornd(1,utility_norm(leader));
        if X==1
           S(leader)=S(leader)+1;
           S_c(index_c)=S_c(index_c)+1;
        else
           F(leader)=F(leader)+1;
           F_c(index_c)=F_c(index_c)+1;
        end
    else
        if leader_1==1
           theta = betarnd(1+S(leader:leader+1),1+F(leader:leader+1));
           [~,index]=max(theta);
           a_UTSC=[a_UTSC leader-1+index];
           r_UTSC=[r_UTSC utility_max-utility(leader-1+index)];
           X = binornd(1,utility_norm(leader-1+index));
           if X==1
              S(leader-1+index)=S(leader-1+index)+1;
              S_c(index_c)=S_c(index_c)+1;
           else
              F(leader-1+index)=F(leader-1+index)+1;
              F_c(index_c)=F_c(index_c)+1;
           end
        elseif leader_1==n_2
           theta = betarnd(1+S((leader-1):leader),1+F((leader-1):leader));
           [~,index]=max(theta);
           a_UTSC=[a_UTSC leader+index-2];
           r_UTSC=[r_UTSC utility_max-utility(leader+index-2)];
           X = binornd(1,utility_norm(leader+index-2));
           if X==1
              S(leader+index-2)=S(leader+index-2)+1;
              S_c(index_c)=S_c(index_c)+1;
           else
              F(leader+index-2)=F(leader+index-2)+1;
              F_c(index_c)=F_c(index_c)+1;
           end
        else
           theta = betarnd(1+S(leader-1:leader+1),1+F(leader-1:leader+1));
           [~,index]=max(theta);
           a_UTSC=[a_UTSC leader+index-2];
           r_UTSC=[r_UTSC utility_max-utility(leader+index-2)];
           X = binornd(1,utility_norm(leader+index-2));
           if X==1
              S(leader+index-2)=S(leader+index-2)+1;
              S_c(index_c)=S_c(index_c)+1;
           else
              F(leader+index-2)=F(leader+index-2)+1;
              F_c(index_c)=F_c(index_c)+1;
           end
        end
    end
end
regret_UTSC=[];%caculate the cumulative regret
for i=1:T
    regret_UTSC=[regret_UTSC sum(r_UTSC(1:i))];
end
regret_avg_UTSC=[regret_avg_UTSC;regret_UTSC];
a_avg_UTSC=[a_avg_UTSC;a_UTSC];
end

%MUTSC algorithm
regret_avg_MUTSC=[];
a_avg_MUTSC=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
S_c=zeros(1,n_1);
F_c=zeros(1,n_1);
a_MUTSC=[];%record the action for TS
r_MUTSC=[];
for i=1:T
    theta_c = betarnd(1+S_c,1+F_c);
    [~,index_c]=max(theta_c);%find cluster
    theta = betarnd(1+S(((index_c-1)*n_2+1):(index_c*n_2)),1+F(((index_c-1)*n_2+1):(index_c*n_2)));
    [~,index]=max(theta);
    action=(index_c-1)*n_2+index;
    a_MUTSC=[a_MUTSC action];
    r_MUTSC=[r_MUTSC utility_max-utility(action)];
    X = binornd(1,utility_norm(action));
    X_1 = binornd(1,probability(action));
    if X==1
        S_c(index_c)=S_c(index_c)+1;
    else
        F_c(index_c)=F_c(index_c)+1;
    end
    if X_1==1
        S(action)=S(action)+1;
    else
        F(action)=F(action)+1;
    end
end
regret_MUTSC=[];%caculate the cumulative regret
for i=1:T
    regret_MUTSC=[regret_MUTSC sum(r_MUTSC(1:i))];
end
regret_avg_MUTSC=[regret_avg_MUTSC;regret_MUTSC];
a_avg_MUTSC=[a_avg_MUTSC;a_MUTSC];
end



set(0,'defaulttextinterpreter','latex'); % allows you to use latex math
set(0,'defaultlinelinewidth',2); % line width is set to 2
set(0,'DefaultLineMarkerSize',10); % marker size is set to 10
set(0,'DefaultTextFontSize', 16); % Font size is set to 16
set(0,'DefaultAxesFontSize',16); % font size for the axes is set to 16
figure(1)
x=1:1:T;
%plot(x,0.98*packet_num_CUUCB(1:1000), '--c',x,packet_num_CUCB(1:1000),'-*',x,packet_num_CUUCB(1:1000),'m-.+',x,0.97*packet_num_CUCB_1(1:1000),'-bo',x,packet_num_CUCB_1(1:1000),'-ro','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(x,1.2*mean(regret_co3(:,1:1000)), '--c',x,mean(regret_co2(:,1:1000)),'-*',x,mean(regret_co33(:,1:1000)),'m-.+',x,mean(regret_co22(:,1:1000)),'-bo','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(X, Y1, '-bo', X, Y2, '{rs', X); % plotting three curves Y1, Y2 for the same X
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,mean(regret_avg_TS),'r--h',x,mean(regret_avg_MTS),'g-*',x,mean(regret_avg_UTS),'-bo',x,mean(regret_avg_MUTS),'c-s',x,mean(regret_avg_UCB),'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
plot(x,mean(regret_avg_TS),'r--h',x,mean(regret_avg_MTS),'g-*',x,mean(regret_avg_TSC),'-bo',x,mean(regret_avg_MTSC),'c-s',x,mean(regret_avg_UTSC),'m-.+',x,mean(regret_avg_MUTSC),'-ro','LineWidth',2,'MarkerIndices',1:5000:length(x));
%plot(x,rate_HUUCB_1,'r--h',x,rate_HUUCB_2,'g-*','LineWidth',2,'MarkerIndices',1:5000:length(x));
grid on; % grid lines on the plot

%legend('HTS(good group)', 'HTS(Partial information)','TS','HTS(kmean)');
%legend('TS','MTS','UTS','MUTS','UCB');
legend('TS','MTS','TSC','MTSC','UTSC','MUTSC');
% ylabel('$Regret$ (Kbps)');
% 
% xlabel('$_$ (frames=sec)');

xlabel('Time Slot');
ylabel('Regret');
title('Regret vs Time');

