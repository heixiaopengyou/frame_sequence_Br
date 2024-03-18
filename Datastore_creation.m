% Dataset link:
% https://drive.google.com/file/d/1bfaEjv7hefHj3jVfstuV1Wc9CEKdtxtD/view?usp=sharing

% Path to the main data folder
main_folder_train = '.../Briareo_dataset_mat30/train.zip/';
main_folder_val = '.../Briareo_dataset_mat30/validation.zip/';
main_folder_test = '.../Briareo_dataset_mat30/test.zip/';
% Initialize cell arrays to store concatenated sequences and labels
sequences_train = {};
labels_train = {};

% Listing all candidate recording folders
candidate_folders = dir(fullfile(main_folder_train, '*')); % candidate_folders = dir(fullfile(main_folder_val, '*')) % candidate_folders = dir(fullfile(main_folder_test, '*'))
%%
% Looping through candidate recording folders
for candidate_idx = 1:numel(candidate_folders)
    candidate_folder = candidate_folders(candidate_idx).name;
    
    % Skip folders that are not participant folders
    if ~isfolder(fullfile(main_folder_train, candidate_folder)) || ismember(candidate_folder, {'.', '..'})
        continue;
    end
    
    % List all participant folders ('000', '001', ...)
    participant_folders = dir(fullfile(main_folder_train, candidate_folder, '*'));
    
    % Looping through participant folders
    for participant_idx = 1:numel(participant_folders)
        participant_folder = participant_folders(participant_idx).name;
        
        % Skipping folders that are not activity folders
        if ~isfolder(fullfile(main_folder_train, candidate_folder, participant_folder)) || ismember(participant_folder, {'.', '..'})
            continue;
        end
        
        % Extracting the class labels ('g00', 'g01', ...)
        class_label = participant_folder;
        
        % Initialize an empty array to store concatenated sequences
        concatenated_sequence = [];
        
        % Listing all frame files in the tof/depth folder
        frame_files = dir(fullfile(main_folder_train, candidate_folder, participant_folder, '**', 'tof', 'depth', '*.mat'));
        
        % Loop through frame files
        for frame_idx = 1:numel(frame_files)
            frame_file = frame_files(frame_idx);
            
            % Load data from .mat file
            mat_data = load(fullfile(frame_file.folder, frame_file.name));
            frame_data = mat_data.arr_0;
            
            % Normalize and resize the frame data
            frame_data = mapminmax(frame_data); % Using MinMax scaling to avoid NaN values
            frame_data = imresize(frame_data, [8, 8], 'bicubic');
            frame_data = cast(frame_data, 'single'); % Coverting to uint8 displayed no information
            
            % Concatenating the frame data to the sequence
            concatenated_sequence = cat(3, concatenated_sequence, frame_data);
            
            % Check if the sequence length reaches 30 frames
            if size(concatenated_sequence, 3) == 30
                % Store the concatenated sequence and corresponding label
                sequences_train{end+1} = concatenated_sequence;
                labels_train{end+1} = class_label;
                
                % Reseting the concatenated sequence for the next batch of 30 frames
                concatenated_sequence = [];
            end
        end
    end
end

%%
% Converting labels to categorical array
labels_train1 = reshape(labels_train, [], 1);
labels_cat_train = categorical(labels_train1);
data_train = arrayDatastore(sequences_train);
% Combining datastores
ds_train = combine(data_train, label_cat_train);
