addpath('preprocess')

parpool('local',32)

%processor = 'CGMM_RLS_MPDR'
processor = 'AuxIVA_DC_SVE';

%% path
root = "/home/data/kbh/CHiME4/merged_WAV/";
root_output = ['/home/data/kbh/CHiME4/' processor '/'];
SNR_dirs=["SNR-7","SNR-5","SNR0","SNR5","SNR7","SNR10"];

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

%%  Processing Loop
tic
for SNR_idx=1:length(SNR_dirs)
    tmp = strcat(root, SNR_dirs(SNR_idx), "/", "noisy","/","*.wav");
    target_list = dir(tmp);

    % dir struct : name,folder,data,bytes,isdir,datenum
    mkdir(strcat(root_output,SNR_dirs(SNR_idx),"/","noisy"));
    mkdir(strcat(root_output,SNR_dirs(SNR_idx),"/","estimated_speech"));
    mkdir(strcat(root_output,SNR_dirs(SNR_idx),"/","estimated_noise"));

    parfor (target_idx = 1:length(target_list),32)
        target = target_list(target_idx);
        input_path = [target.folder  '/'  target.name];
        output_path = strcat(root_output,"/",SNR_dirs(SNR_idx),"/");

        x = audioread(input_path);

        if processor == 'CGMM_RPS_MPDR'
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
        % normalize
        estimated_noise= estimated_noise/max(abs(estimated_noise));
        estimated_speech = estimated_speech/max(abs(estimated_speech));

        % save
        audiowrite(strcat(output_path,'noisy','/',target.name),noisy(:,1),fs);
        audiowrite(strcat(output_path,'estimated_speech','/',target.name),estimated_speech,fs);
        audiowrite(strcat(output_path,'estimated_noise','/',target.name),estimated_noise,fs);
            
        %disp(['progress ' num2str(SNR_idx) '/'  num2str(length(SNR_dirs)) ' | ' num2str(target_idx) '/' num2str(length(target_list))])
    end
end

toc