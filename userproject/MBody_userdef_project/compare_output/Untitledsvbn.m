close all
Array_lol=csvread('compare.out.st');
t=Array_lol(:, 1);
v=Array_lol(:, 2);
figure
% scatter(t,v,'b','.')
for i=1:numel(t)
    if (v(i)>-1 && v(i)<100)
        scatter(t(i),v(i),'r','.')
    elseif (v(i)>99 && v(i)<1100)
        scatter(t(i),v(i),'b','.')
    elseif (v(i)>1099 && v(i)<1120)
        scatter(t(i),v(i),'g','.')
    else
        scatter(t(i),v(i),'k','.')
    end
    hold on
end
hold on;
h = zeros(4, 1);
h(1) = plot(0,0,'.r', 'visible', 'off');
h(2) = plot(0,0,'.b', 'visible', 'off');
h(3) = plot(0,0,'.g', 'visible', 'off');
h(4) = plot(0,0,'.k', 'visible', 'off');
legend(h, {'PNs','KCs','LHIs','DNs'},'FontSize',12);
refline(0,100)
refline(0,1100)
refline(0,1120)
refline(0,1220)
ylim([0 1400])
xlabel('Time (ms)')
ylabel('# neuron')
set(gca,'box','off')
set(gcf,'PaperPositionMode','auto');
pos=get(gcf,'pos');
set(gcf,'PaperSize',[pos(3), pos(4)]);
print(gcf,'rasterplot.jpeg','-djpeg','-r600');