Array_adex=csvread('outname.out.Vm');
t_adex=Array_adex(:, 1);
v_adex=Array_adex(:, 2:21);
plot(t_adex,v_adex,'r')
