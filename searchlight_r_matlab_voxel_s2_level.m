% Foldername: custom_python
% Filename: searchlight_r_matlab_voxel_s2_level.m
% Reference: existing level-based searchlight version; not the trial-wise input generator used by the supplied first-level GLMs.
package_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(package_dir, 'toolbox_3rdParty'));
subject_list = strings(1, 30);
for i = 1:30
    subject_list(i) = sprintf('par%02d', i);
end

cur_pe_keyword = 'level';

for ith_sess = 2
    % ============================================================
    % 从 design.fsf 提取 evtitle -> PE 映射表 (每个被试一张)
    % ============================================================
    fsf_root = [[package_dir, filesep, 'fsf_feat_Files_s'],num2str(ith_sess),'_v2/'];
    ev_tables = cell(length(subject_list), 1);   % 每个被试存一个 table

    for ith_sub = 1:length(subject_list)
        cur_sub = char(subject_list(ith_sub));

        fid = fopen([fsf_root, 'fsf_par',num2str(ith_sub),'_s',num2str(ith_sess),'.fsf'], 'r');
        raw = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        lines = raw{1};

        % 匹配 set fmri(evtitleN) "name"
        tok = regexp(lines, 'set fmri\(evtitle(\d+)\)\s+"(.+)"', 'tokens');
        hit = ~cellfun(@isempty, tok);
        tok = tok(hit);

        n_ev = numel(tok);
        pe_num   = zeros(n_ev, 1);
        ev_name  = cell(n_ev, 1);
        for k = 1:n_ev
            pe_num(k)  = str2double(tok{k}{1}{1});   % evtitle 编号 = PE 编号
            ev_name{k} = tok{k}{1}{2};               % e.g. "beta_clickbait_1"
        end

        % 按 PE 编号排序 (应该已经有序, 保险起见)
        [pe_num, si] = sort(pe_num);
        ev_name = ev_name(si);

        % 构建 pe 标签: pe1, pe2, ...
        pe_label = arrayfun(@(x) sprintf('pe%d', x), pe_num, 'UniformOutput', false);

        ev_tables{ith_sub} = table(pe_num, pe_label, ev_name, ...
            'VariableNames', {'pe_num', 'pe_label', 'ev_name'});
    end


    %%
    output_path = [[package_dir, filesep, 'file_generated/searchlight_r_rsa_s'], num2str(ith_sess), '_level_subj/'];
% Use the existing output directory; do not create folders.
    mask_wb = MRIread(fullfile(package_dir, 'brainmask', 'mask_wholebrain.nii'));
    mask_wb = mask_wb.vol;
    [fir_di, sec_di, thi_di] = size(mask_wb);
    [d1, d2, d3] = ind2sub([fir_di, sec_di, thi_di], find(mask_wb == 1));
    voxel_array = [d1, d2, d3];
    count_voxel = size(voxel_array, 1);

    spherec_coordinate = sphericalRelativeRoi(6, [2 2 2]);
    n_sphere = size(spherec_coordinate, 1);

    mask_wb = MRIread(fullfile(package_dir, 'brainmask', 'mask_wholebrain.nii'));
    [d1,d2,d3] = ind2sub(size(mask_wb.vol), find(mask_wb.vol==1));


    bar_width = 20;
    fprintf('Searchlight [%s] %2d/%d', repmat(' ', 1, bar_width), 0, length(subject_list));
    for ith_sub = 1:length(subject_list)
        cur_sub = subject_list(ith_sub);

        cur_path = [package_dir, filesep, 'file_generated/fsl_output_rsa_level_v2_'];
        dir_keyfile = dir(fullfile([cur_path, char(cur_sub), '_s', num2str(ith_sess), '.feat/stats/standard*']));
        beta_list = strcat([cur_path, char(cur_sub), '_s', num2str(ith_sess), '.feat/stats/'], {dir_keyfile.name}');

        pe_list = zeros(length(beta_list), 1);
        for ith_pe = 1:length(beta_list)
            pe_list(ith_pe) = str2double(cell2mat(extractBetween(beta_list{ith_pe}, 'pe', '.nii')));
        end
        [~, sort_index] = sort(pe_list);
        beta_list = beta_list(sort_index);

        %
        cur_ev_table = ev_tables{ith_sub};
        cur_ev_idx = cellfun(@(x) contains(x, cur_pe_keyword), cur_ev_table(:,3).ev_name);
        cur_ev_list = cur_ev_table(find(cur_ev_idx), :);
        beta_list = beta_list(cur_ev_list.pe_num);


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
end
