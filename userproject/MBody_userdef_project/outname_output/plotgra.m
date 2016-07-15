close all
Array_lol=csvread('outname.out.st');
t=Array_lol(:, 1);
v=Array_lol(:, 2);
figure
scatter(t,v,'b','.')
% for i=1:numel(t)
%     if (v(i)>-1 && v(i)<100)
%         scatter(t(i),v(i),'r','.')
%     elseif (v(i)>99 && v(i)<1100)
%         scatter(t(i),v(i),'b','.')
%     elseif (v(i)>1099 && v(i)<1120)
%         scatter(t(i),v(i),'g','.')
%     else
%         scatter(t(i),v(i),'k','.')
%     end
%     hold on
% end
refline(0,100)
refline(0,1100)
refline(0,1120)
refline(0,1220)
ylim([0 1400])
set(gca,'box','off')
set(gcf,'PaperPositionMode','auto');
pos=get(gcf,'pos');
set(gcf,'PaperSize',[pos(3), pos(4)]);
% print(gcf,'rasterplot.jpeg','-djpeg','-r600');