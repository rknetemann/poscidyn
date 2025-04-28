clc
clear
X=dlmread ('tot1.txt');
% bifs=dlmread('bifs.txt')
hold on
k=1;
j=1;
for k=1:length(X)
    if X(k,2)<0 | X(k,2)>0
        x5(k,j)=(X(k,5));
        x7(k,j)=(X(k,7));
        k=k+1;
        if k+1>length(X)
            break
        else
            if X(k,2)*X(k+1,2)<0
                j=j+1;
            end
        end
    end
end
for i=1:j
    if mod(i,2)==0
           plot(nonzeros((x5(:,i)))/1.000,(nonzeros(x7(:,i))),'--k','linewidth',2)
    else
         plot(nonzeros((x5(:,i)))/1.000,(nonzeros(x7(:,i))),'k','linewidth',2)
    end
    hold on
end
% get(gcf,'CurrentAxes');
% set(gca,'FontName','times','FontSize',18,'fontweight','b','LineWidth',4.5);
% % xlabel('\Omega/\omega','fontsize',18,'fontweight','b','fontname','times','fontangle','italic')
% % ylabel('Displacement','fontsize',18,'fontweight','b','fontname','times','fontangle','italic')
%  xlabel('frequency ratio r','fontsize',18,'fontweight','b','fontname','times','fontangle','italic')
%  ylabel('Dimensionless amplitude','fontsize',18,'fontweight','b','fontname','times','fontangle','italic')
% % % % plot(bifs(:,2),bifs(:,7),'*k')


%