addpath('preprocess')

parpool('local',22)

processor = 'CGMM_RLS_MPDR'
%processor = 'AuxIVA_DC_SVE';

%% path
root = "/home/data2/kbh/CHiME4_eval/";
%root_output = ['/home/data/kbh/CHiME4/' processor '/'];
root_output = ['/home/data2/kbh/CHiME4/' processor ];
%SNR_dirs=["SNR-7","SNR-5","SNR0","SNR5","SNR7","SNR10"];
SNR_dirs=["SNR10"];

%% params
winL = 1024;
nch=6;
fs=16000;

%% params : CGMM
gamma = 0.99;
Ln = 5;
MVDR_on = 0;

%% params : ICA
pdf_opt = 2;
mdp_opt = 1;
online_opt = 0;

mkdir(root_output);

%%  Processing Loop
tic
target_list = dir([root , '/**/*.wav']);

% dir struct : name,folder,data,bytes,isdir,datenum
mkdir(strcat(root_output,"/","noisy"));
mkdir(strcat(root_output,"/","estim"));
mkdir(strcat(root_output,"/","noise"));

list_dir1 = [
    "noisy"
    "estim"
    "noise"
    ];

list_dir2 = [
    "dt05_bus_real"
    "dt05_caf_real"
    "dt05_ped_real"
    "dt05_str_real"
    "dt05_bus_simu"
    "dt05_caf_simu"
    "dt05_ped_simu"
    "dt05_str_simu"
    "et05_bus_real"
    "et05_caf_real"
    "et05_ped_real"
    "et05_str_real"
    "et05_bus_simu"
    "et05_caf_simu"
    "et05_ped_simu"
    "et05_str_simu"
    ]

for i_1 = 1:length(list_dir1)
    for i_2 = 1:length(list_dir2)
        mkdir(strcat(root_output,"/",list_dir1(i_1),"/",list_dir2(i_2)))
    end
end

parfor (target_idx = 1:length(target_list),32)
    target = target_list(target_idx);

    dir = char(split(target.folder,'/'))
    input_path = [target.folder  '/'  target.name];
    output_path = strcat(root_output,"/",dir,"/");

    x = audioread(input_path);

    if processor == 'CGMM_RLS_MPDR'
        [estimated_speech,estimated_noise] = CGMM_RLS_tuning(x,winL,gamma,Ln,MVDR_on);

        % sync
        %[r,lags] = xcorr(x(:,1),estimated_speech(:,1));   
        %[max_val,max_idx] = max(abs(r));
        %delay = lags(max_idx);
        %disp(delay)

        % shift size delay
        delay = -768;
        if delay > 0
            sync = x(delay+1 : end,1);
        else
            pad = zeros(abs(delay),1);
            sync = cat(1,pad,x(:,1));
        end
        noisy= sync(1:length(estimated_speech),1);

    elseif processor == 'AuxIVA_DC_SVE' 
        [estimated_speech,estimated_noise] = run_AuxIVA_DC_SVE(x,winL,pdf_opt,mdp_opt,online_opt);
        noisy = x;

    end
    %% normalize
    %estimated_noise= estimated_noise/max(abs(estimated_noise));
    %estimated_speech = estimated_speech/max(abs(estimated_speech));

    %% normalization based on input scale
    estimated_speech = estimated_speech/max(abs(noisy(:,1)));
    estimated_noise = estimated_noise/max(abs(noisy(:,1)));


    % save
    audiowrite(strcat(output_path,'noisy','/',dir,'/',target.name),noisy(:,1),fs);
    audiowrite(strcat(output_path,'estimated_speech','/',dir,'/',target.name),estimated_speech(:,1),fs);
    audiowrite(strcat(output_path,'estimated_noise','/',dir,'/',target.name),estimated_noise(:,1),fs);
        
    %disp(['progress ' num2str(SNR_idx) '/'  num2str(length(SNR_dirs)) ' | ' num2str(target_idx) '/' num2str(length(target_list))])
end

toc