% Dataset link:
% https://drive.google.com/file/d/1bfaEjv7hefHj3jVfstuV1Wc9CEKdtxtD/view?usp=sharing

% Path to the main data folders
main_folder_train = '.../Briareo_dataset_mat30/train.zip/';
main_folder_val = '.../Briareo_dataset_mat30/validation.zip/';
main_folder_test = '.../Briareo_dataset_mat30/test.zip/';
%%
% Creating a sample datastore (test_ds)
sequences_test = {};
labels_test = {};

% Listing all candidate recording folders
candidate_folders = dir(fullfile(main_folder_test, '*'));

% Looping through candidate recording folders
for candidate_idx = 1:numel(candidate_folders)
    candidate_folder = candidate_folders(candidate_idx).name;
    
    % Skipping folders that are not participant folders (e.g., '.', '..')
    if ~isfolder(fullfile(main_folder_test, candidate_folder)) || ismember(candidate_folder, {'.', '..'})
        continue;
    end
    
    % Listing all participant folders (e.g., '000', '001', ...)
    participant_folders = dir(fullfile(main_folder_test, candidate_folder, '*'));
    
    % Loop through participant folders
    for participant_idx = 1:numel(participant_folders)
        participant_folder = participant_folders(participant_idx).name;
        
        % Skipping folders that are not activity folders
        if ~isfolder(fullfile(main_folder_test, candidate_folder, participant_folder)) || ismember(participant_folder, {'.', '..'})
            continue;
        end
        
        % Extracting the class label (e.g., 'g00', 'g01', ...)
        class_label = participant_folder;
        
        % Initializing an empty array to store concatenated sequences
        concatenated_sequence = [];
        
        % Listing all frame files in the tof/depth folder
        frame_files = dir(fullfile(main_folder_test, candidate_folder, participant_folder, '**', 'tof', 'depth', '*.mat'));
        
        % Looping through frame files
        for frame_idx = 1:numel(frame_files)
            frame_file = frame_files(frame_idx);
            
            % Loading data frames from .mat file
            mat_data = load(fullfile(frame_file.folder, frame_file.name));
            frame_data = mat_data.arr_0;
            
            % Normalizing and resize the frame data (Standardizing)
            frame_data = mapminmax(frame_data); % Normalize() function generates NaN values
            frame_data = imresize(frame_data, [8, 8], 'bicubic'); % Resampling the frames to meet the required shape of the sensor's output
            frame_data = cast(frame_data, 'single'); % uint8 type displayed no information in the frames
            
            % Concatenating the frame data to the sequence
            concatenated_sequence = cat(3, concatenated_sequence, frame_data);
            
            % Checking if the sequence length reaches 30 frames
            if size(concatenated_sequence, 3) == 30
                % Store the concatenated sequence and corresponding label
                sequences_test{end+1} = concatenated_sequence;
                labels_test{end+1} = class_label;
                
                % Reseting the concatenated sequence for the next batch of 30 frames
                concatenated_sequence = [];
            end
        end
    end
end

%%
sequences_test = reshape(sequences_test, [], 1); % Reshaping the concatenated sequences to N x 1
data = arrayDatastore(sequences_test, 'ReadSize',1); % Creating a datastore from the sequences
labels_test = reshape(labels_test, [], 1); % Reshaping the Labels 
labels_cat_test = categorical(labels_test); % Converting the labels to categorical
lab = arrayDatastore(labels_cat_test); % Creating a datastore from the categorical labels
ds_test = combine(data,lab); % Combining both datasets
