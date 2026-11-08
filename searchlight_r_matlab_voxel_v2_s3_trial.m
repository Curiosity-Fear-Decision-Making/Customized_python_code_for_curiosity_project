% Foldername: custom_python
% Filename: searchlight_r_matlab_voxel_v2_s3_trial.m
package_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(package_dir, 'toolbox_3rdParty'));
subject_list = strings(1, 30);
for i = 1:30
    subject_list(i) = sprintf('par%02d', i);
end


%%
cur_period = 'clickbaitshockstick'; 
ith_sess = 3;

output_path = [[package_dir, filesep, 'file_generated/searchlight_r_rsa_s'], num2str(ith_sess), '_', cur_period , '/'];
% Use the existing output directory; do not create folders.
mask_wb = MRIread(fullfile(package_dir, 'brainmask', 'mask_wholebrain.nii'));
mask_wb = mask_wb.vol;
[fir_di, sec_di, thi_di] = size(mask_wb);
[d1, d2, d3] = ind2sub([fir_di, sec_di, thi_di], find(mask_wb == 1));
voxel_array = [d1, d2, d3];
count_voxel = size(voxel_array, 1);

spherec_coordinate = sphericalRelativeRoi(6, [2 2 2]);
n_sphere = size(spherec_coordinate, 1);

% save voxel mapping once per session
% save(fullfile(output_path, 'voxel_mapping.mat'), 'voxel_array', '-v7.3');

bar_width = 20;
fprintf('Searchlight [%s] %2d/%d', repmat(' ', 1, bar_width), 0, length(subject_list));
for ith_sub = 1:length(subject_list)
    cur_sub = subject_list(ith_sub);

    cur_path = [[package_dir, filesep, 'file_generated/fsl_output_trial_'],cur_period,'_'];
    dir_keyfile = dir(fullfile([cur_path, char(cur_sub), '_s', num2str(ith_sess), '.feat/stats/standard*']));
    beta_list = strcat([cur_path, char(cur_sub), '_s', num2str(ith_sess), '.feat/stats/'], {dir_keyfile.name}');

    pe_list = zeros(length(beta_list), 1);
    for ith_pe = 1:length(beta_list)
        pe_list(ith_pe) = str2double(cell2mat(extractBetween(beta_list{ith_pe}, 'pe', '.nii')));
    end
    [~, sort_index] = sort(pe_list);
    beta_list = beta_list(sort_index);
    beta_list = beta_list(1:128);


    num_betas = length(beta_list);

    % load all beta volumes
    beta_volumes = cell(num_betas, 1);
    for n = 1:num_betas
        tmp = MRIread(beta_list{n});
        beta_volumes{n} = tmp.vol;
    end

    % stack into 2D for faster linear indexing: [X*Y*Z, num_betas]
    vol_size = [fir_di, sec_di, thi_di];
    beta_2d = zeros(prod(vol_size), num_betas, 'single');
    for n = 1:num_betas
        beta_2d(:, n) = single(beta_volumes{n}(:));
    end
    clear beta_volumes;

    % upper-triangle indices (excluding diagonal)
    triu_mask = triu(true(num_betas), 1);
    n_pairs = sum(triu_mask(:));

    % output: [n_pairs x count_voxel] in single, will quantize to int16 at the end
    corr_2d = zeros(n_pairs, count_voxel, 'single');

    % Constant broadcast to workers
    beta_const = parallel.pool.Constant(beta_2d);
    sphere_const = parallel.pool.Constant(spherec_coordinate);
    triu_const = parallel.pool.Constant(triu_mask);

    parfor voxel_index = 1:count_voxel
        try
            spherec = sphere_const.Value + voxel_array(voxel_index, :);
            oob = spherec(:,1) < 1 | spherec(:,2) < 1 | spherec(:,3) < 1 | ...
                spherec(:,1) > fir_di | spherec(:,2) > sec_di | spherec(:,3) > thi_di;
            spherec(oob, :) = [];

            if isempty(spherec)
                corr_2d(:, voxel_index) = 0;
                continue;
            end

            lin_idx = sub2ind(vol_size, spherec(:,1), spherec(:,2), spherec(:,3));
            eachspherec_by_trial = beta_const.Value(lin_idx, :);

            % drop voxels that are NaN or zero in first beta
            bad = isnan(eachspherec_by_trial(:,1)) | eachspherec_by_trial(:,1) == 0;
            eachspherec_by_trial(bad, :) = [];

            if size(eachspherec_by_trial, 1) >= 5
                rho = corr(eachspherec_by_trial, 'Type', 'Pearson');
                t_rho = 0.5 * log((1 + rho) ./ (1 - rho));
                t_rho(isinf(t_rho) | isnan(t_rho)) = 0;
                tm = triu_const.Value;
                corr_2d(:, voxel_index) = t_rho(tm);
            else
                corr_2d(:, voxel_index) = 0;
            end

        catch ME
            fprintf('voxel %d error: %s\n', voxel_index, ME.message);
        end
    end

    % quantize to int16: scale by 10000 (precision ~0.0001, range ±3.27)
    scale = 10000;
    corr_scaled = corr_2d * scale;
    corr_scaled(corr_scaled >  32767) =  32767;
    corr_scaled(corr_scaled < -32768) = -32768;
    corr_int16 = int16(round(corr_scaled));
    clear corr_2d corr_scaled beta_2d beta_const sphere_const triu_const;

    % save as .npy (fast loading in Python)
    file_name = fullfile(output_path, [char(cur_sub), '_run', num2str(ith_sess), '.npy']);
    writeNPY(corr_int16, file_name);

    % save meta info once per session (scale, num_betas, triu_mask)
    if ith_sub == 1
        meta_file = fullfile(output_path, 'meta.mat');
        save(meta_file, 'scale', 'num_betas', 'triu_mask', '-v7');
    end

    filled = floor(bar_width * ith_sub / length(subject_list));
    fprintf('\rSearchlight [%s%s] %2d/%d', repmat('=', 1, filled), ...
        repmat(' ', 1, bar_width - filled), ith_sub, length(subject_list));
    if ith_sub == length(subject_list)
        fprintf('\n');
    end
    clear corr_int16;
end
