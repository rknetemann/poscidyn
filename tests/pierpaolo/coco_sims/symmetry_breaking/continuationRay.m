%% clean 
close all; clear all; clc;

%% TIMING START
tStart = tic;

% % load coco functions
cd '/usr/local/coco_2025January28/coco_2025January28/'; 
startup;
% 
% % go to working dir
cd '/home/raymo/Projects/parameter-identification-nanomechanical-resonators/tests/pierpaolo/coco_sims/symmetry_breaking';

%% TRAJECTORY  1
 
p0 = [0.03; 0.5; 0.00]; %  fx, omega, fy
[~, x0] = ode45(@(t,x) vec_fields_Coupled2DOF_aut(x,p0), [0 26000], [0;0.01; 0;0; 0;1]); % Transients
% B - periodic orbit
options = odeset( 'abstol',10^-12,'reltol',10^-12,'Stats','on');%,'MaxStep',1e-2
[t0, x0] = ode45(@(t,x) vec_fields_Coupled2DOF_aut(x,p0), [0 2*pi/0.5 ], x0(end,:)',options); % Approximate periodic orbit 

% Ap - transient
figure(1);clf; hold on; set(gcf,'color','w');    grid on; hold on;
box on;
strFontWeight = 'bold';
set(gca,'FontSize',15)
H=gca;
H.LineWidth=1;
hold on;
scatter(x0(1,1),x0(1,2),'m','linewidth',1)
plot(x0(:,1),x0(:,2))

xlabel('position $q_1$','Interpreter','latex','FontWeight', strFontWeight);
ylabel('velocity $q_2$','Interpreter','latex','FontWeight', strFontWeight);
set(gcf,'position',[10,10,500,500])

%% CHECK INITIAL VECTOR
x0(1,:)
x0(end,:)

%% C - continuation
prob = coco_prob();
% prob=coco_set(prob,'cont','h_max',0.001,'PtMX',1500,'NPR',500);%,'h_max',0.01,'h0',0.001,'NPR',50);,'PtMX',1500,'ItMX',2000,'NPR',1

prob = coco_set(prob, 'ode', 'autonomous', true);
funcs = {@vec_fields_Coupled2DOF_aut};%, @bistable_dx, @bistable_dp};
coll_args = [funcs, {t0, x0, {'fx' 'omega' 'fy'  }, p0}];
prob = ode_isol2po(prob, '', coll_args{:});
 
% prob = po_mult_add(prob, 'po.orb');  % Store Floquet multipliers with bifurcation data
% cont_args = {1, {'po.period' 'T'  }, [1*pi/1.15 2*pi/0.95]};
cont_args = {1, {  'omega'  'fx'}, [ 0.5  1.5 ]};
% prob=coco_set(prob,'po.orb.coll','NTST',20,'NCOL',10);
prob = coco_set(prob, 'cont', 'PtMX',200,'NPR',1000);
% prob = coco_set(prob, 'cont', 'NAdapt',1,'PtMX',1500,'ItMX',1500 , 'PtMX',200 ,'NPR',100);
 
fprintf('\n Run=''%s'': Continue primary family of periodic orbits.\n','freq_resp');

bd1_low  = coco(prob, 'freq_resp_TRY1', [], cont_args{:});

%% CA - continuation in amplitude
prob = coco_prob();

labs = coco_bd_labs(bd1_low, 'EP');
prob = ode_po2po(prob, '', 'freq_resp_TRY1', labs(1));

cont_args ={1, { 'fx' 'omega' }, [0.03 0.3]};

prob = coco_add_event(prob, 'UZ', 'fx', [0.03, 0.057, 0.084, 0.111, 0.138, 0.165, 0.192, 0.219, 0.246, 0.273, 0.3]);
 
bd2_low  = coco(prob, 'freq_resp_TRY2', [], cont_args{:});

%% multiple FRC
labs  = coco_bd_labs(bd2_low, 'UZ');
for lab= labs
    
    prob = coco_prob();
    prob = ode_po2po(prob, '', 'freq_resp_TRY2', lab);
   
    cont_args = {1, {'omega' 'fx'}, [0.5 1.5]};
    prob = coco_set(prob, 'cont', 'NAdapt',1,'PtMX',300,'NTST',40,'NPR',1000);
    fprintf(...
        '\n Run=''%s'': Continue family of periodic orbits from point %d in run ''%s''.\n', ...
        sprintf('lab=%d', lab), lab, 'fx');
    
    coco(prob, sprintf('lab=%d', lab), [], cont_args{:});
end

%% PLOT
figure(243); clf; set(gcf,'color','w'); title('Sims Raymond COCO version'); hold on; grid on; box on
box on;
strFontWeight = 'bold';
set(gca,'FontSize',15)
H=gca;
H.LineWidth=1;
hold on;

grid on
set(gcf,'position',[10,10,500,500])
 
set(gcf,'color','white')
set(gca, 'FontName', 'Arial')
%
 
thm = struct( 'special', {{'PD','SN'}});
  
for lab=labs
    coco_plot_bd( sprintf('lab=%d', lab), 'omega',  'MAX(x)' );
end
% coco_plot_bd(thm,'freq_resp_TRY1', 'omega' ,  'MAX(x)')

%% TIMING END
totalTime = toc(tStart);
fprintf('\nTotal script runtime: %.3f seconds.\n', totalTime);
