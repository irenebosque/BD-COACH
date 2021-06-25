alpha = 0.8;
tau1 = 0.00045;
tau2 = 0.0007;
tau3 = 0.0003;
tau4 = 0.0002;
time_steps = 20000;
time_steps_vector = 0:1:time_steps;

P_h1 = alpha *exp(-tau1 * time_steps_vector)*100;
P_h2 = alpha *exp(-tau2 * time_steps_vector);
P_h3 = alpha *exp(-tau3 * time_steps_vector);
P_h4 = alpha *exp(-tau4 * time_steps_vector);




plot(time_steps_vector,P_h1, 'LineWidth',2, 'Color','r')
%hold on 

%plot(time_steps_vector,P_h2, 'LineWidth',2, 'Color','b')
%plot(time_steps_vector,P_h3, 'LineWidth',2, 'Color','m')
%plot(time_steps_vector,P_h4, 'LineWidth',2, 'Color','g')



%legend('tau = 0.00045','tau = 0.0007','tau = 0.0003', 'tau = 0.0002')
legend('tau = 0.00045')


title('Probability of giving feedback at each time step')
xlabel('Time steps')
ylabel('% of feedback')

xlim([0 time_steps])


xlabel('Time (in timesteps)')
ylabel('Probability')
ytickformat('percentage')

xlim([0 time_steps])