classdef CustomDatastore < matlab.io.Datastore
    
    properties (Access = private) %I decided to create a custom datastore 
        % Because when I tried debuging I realized the model wasn't able to access the data stored in cells of an array
        % Since each cell was a 8x8x30 tensor
        % With this Datastore, I tried custom training, It worled for one epoch then I realized I had issues on how the data ws loaded.
        CurrentFileIndex double
        FileSet matlab.io.datastore.DsFileSet
        Labels cell
        rootDir char
    end
    
    methods
        function obj = CustomDatastore(rootDir)
            obj.FileSet = matlab.io.datastore.DsFileSet(rootDir, ...
                'IncludeSubfolders', true, ...
                'FileExtensions', '.mat');
            obj.CurrentFileIndex = 1;
            obj.Labels = {};
            obj.rootDir = rootDir;
            
            % Extract labels from folder names
            folders = dir(rootDir);
            folders = folders([folders.isdir]);
            % disp(folders)
            folderNames = {folders(3:end).name}; % Exclude '.' and '..'
            obj.Labels = cellfun(@(x) extractLabel(fullfile(rootDir, x)), folderNames, 'UniformOutput', false);
        end
        
        function tf = hasdata(obj)
            tf = hasfile(obj.FileSet);
        end
        
        function [data, label] = read(obj)
            if ~hasdata(obj)
                error('No more data to read.');
            end
            fileInfo = nextfile(obj.FileSet);
            matData = load(fileInfo.FileName);
            frame_data = matData.arr_0;
            frame_data = mapminmax(frame_data);
            frame_data = imresize(frame_data, [8, 8], 'bicubic');
            % Initialize an array to store concatenated frames
            concatenated_frames = []; % I'll change this line because I get more than 30 frames, which is not supposed to be, I tried with zeros too, but still to find the bug
            
            candidate_folders = dir(fullfile(obj.rootDir, '*'));
            for candidate_idx = 1:numel(candidate_folders)
                candidate_folder = candidate_folders(candidate_idx).name;
                if ~isfolder(fullfile(obj.rootDir, candidate_folder)) || ismember(candidate_folder, {'.', '..'})
                    continue;
                end
                participant_folders = dir(fullfile(obj.rootDir, candidate_folder, '*'));
                for participant_idx = 1:numel(participant_folders)
                    participant_folder = participant_folders(participant_idx).name;
                    if ~isfolder(fullfile(obj.rootDir, candidate_folder, participant_folder)) || ismember(participant_folder, {'.', '..'})
                        continue;
                    end
                    % obj.Labels = participant_folder; % An alternative way to retrieve the labels, but the Labels type should be changed to string I think
                    % concatenated_frames = zeros([size(frame_data), 30], 'single');
                    frame_files = dir(fullfile(obj.rootDir, candidate_folder, participant_folder, '**', 'tof', 'depth', '*.mat'));
                    for frame_idx = 1:min(30, numel(frame_files))
                        frame_file = frame_files(frame_idx);
                        mat_data = load(fullfile(frame_file.folder, frame_file.name));
                        frame_data = mat_data.arr_0;
                        frame_data = mapminmax(frame_data);
                        frame_data = imresize(frame_data, [8, 8], 'bicubic');
                        frame_data = cast(frame_data, 'single');
                        concatenated_frames = cat(3, concatenated_frames, frame_data); % I'm supposed to have a 8x8x30, but the shape is inconsistent, I'll fix it
                    end
                    break;
                end
            end
            label = obj.Labels(obj.CurrentFileIndex);
            % label = obj.Labels{mod(obj.CurrentFileIndex -1, numel(obj.Labels)) + 1};
            % info.Filename = fileInfo.FileName;
            obj.CurrentFileIndex = obj.CurrentFileIndex + 1;
            % Update the shape of 'data' to include the concatenated frames
            data = concatenated_frames;
            % obj.CurrentFileIndex = obj.CurrentFileIndex + 1;
        end
        function reset(obj) % Reset function
            reset(obj.FileSet);
            obj.CurrentFileIndex = 1;
        end
    end
    methods (Hidden = true)
        function frac = progress(obj)
            if hasdata(obj) 
               frac = (obj.CurrentFileIndex-1)/numel(obj.Labels); 
            else 
               frac = 1;  
            end 
        end
    end

    methods (Access = protected)
        function dscopy = copyElement(ds)
            dscopy = copyElement@matlab.mixin.Copyable(ds);
            dscopy.FileSet = copy(ds.FileSet);
        end                
    end
end
function label = extractLabel(folderPath) % Function I use to extract the labels
    % Limit recursion depth to avoid excessive memory usage
    maxDepth = 10;  % Set maximum recursion depth
    label = '';     % Initialize label
    
    % Get a list of all subfolders in the current folder
    folders = dir(folderPath);
    
    % Loop through each subfolder
    for i = 1:numel(folders)
        % Skip '.' and '..' folders
        if isequal(folders(i).name, '.') || isequal(folders(i).name, '..')
            continue;
        end
        
        % If the current subfolder starts with 'g', it's the label
        if startsWith(folders(i).name, 'g')
            label = folders(i).name;
            return; % Exit the function once the label is found
        end
        
        % If the current subfolder is a directory and within recursion depth,
        % recursively search it
        if folders(i).isdir && maxDepth > 0
            label = extractLabel(fullfile(folderPath, folders(i).name));
            if ~isempty(label)
                return; % Exit the function if the label is found
            end
        end
    end
end
