Array_adex=csvread('compare.out.Vm');
t_adex=Array_adex(:, 1);
v_adex=Array_adex(:, 2:21);
d=8;
plot(t_hh,v_hh(:,d),t_adex,v_adex(:,d))
legend({'HH','AdEx'},'FontSize',12)
legend('boxoff')
xlabel('Time (ms)')
ylabel('Voltage (mV)')
% ylim([-85 -35])
% hold on
% plot(t_adex,v_adex(:,d),'r','LineWidth',2)
set(gca,'box','off')
set(gcf,'PaperPositionMode','auto');
pos=get(gcf,'pos');
set(gcf,'PaperSize',[pos(3), pos(4)]);
print(gcf,'Mbody_compare_HHvsAdEx_deviations2.jpeg','-djpeg','-r600');