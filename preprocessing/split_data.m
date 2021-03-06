% Splitting data based on time. HavsDataset has 
% spectrograms arranged in a chronological order
% Thereby each split for a specific class will 
% result in examples taken at different times

load('havsdata.mat')
labels = [];

% Assigning numbers to each label
running_int_label = 1;
vehicle_int_label = 2;
walking_int_label = 3;
clutter_int_label = 4;
two_walking_int_label = 5;
sphere_int_label = 6;

% Converting string labels to integer labels
for i=1:length(HAVSDatasetStruct)
    label = HAVSDatasetStruct(i).Label;
    if strcmp(label, 'running')
        labels = [labels running_int_label];
    elseif strcmp(label, 'vehicle')
        labels = [labels vehicle_int_label];
    elseif strcmp(label, 'walking')
        labels = [labels walking_int_label];
    elseif strcmp(label, 'clutter/noise')
        labels = [labels clutter_int_label];
    elseif strcmp(label, '2_walking')
        labels = [labels two_walking_int_label];
    elseif strcmp(label, 'sphere_swing')
        labels = [labels sphere_int_label];
    end
end

% Finding indexes of examples with a particular label
running_idxs = find(labels == running_int_label);
vehicle_idxs = find(labels == vehicle_int_label);
walking_idxs = find(labels == walking_int_label);
clutter_idxs = find(labels == clutter_int_label);
two_walking_idxs = find(labels == two_walking_int_label);
sphere_idxs = find(labels == sphere_int_label);

% Specify the splitting percentage criteria
train_fraction = 0.7;
validation_fraction = 0.15;
test_fraction = 0.15;

% The running and walking examples need to be combined based on time
% Robert's suggestion
running_idxs_1 = running_idxs(1:2268);
running_idxs_2 = running_idxs(2269:end);

running_count_1 = length(running_idxs_1);
train_running_idxs_1 = running_idxs_1(1:fix(running_count_1*train_fraction));
validation_running_idxs_1 = running_idxs_1(fix(running_count_1*train_fraction)+1:fix(running_count_1*(train_fraction+validation_fraction)));
test_running_idxs_1 = running_idxs_1(fix(running_count_1*(train_fraction+validation_fraction))+1:end);

running_count_2 = length(running_idxs_2);
train_running_idxs_2 = running_idxs_2(1:fix(running_count_2*train_fraction));
validation_running_idxs_2 = running_idxs_2(fix(running_count_2*train_fraction)+1:fix(running_count_2*(train_fraction+validation_fraction)));
test_running_idxs_2 = running_idxs_2(fix(running_count_2*(train_fraction+validation_fraction))+1:end);

walking_idxs_1 = walking_idxs(1:2915);
walking_idxs_2 = walking_idxs(2916:3039);
walking_idxs_3 = walking_idxs(3040:end);

%walking_idxs = flip(walking_idxs);
walking_count_1 = length(walking_idxs_1);
train_walking_idxs_1 = walking_idxs_1(1:fix(walking_count_1*train_fraction));
validation_walking_idxs_1 = walking_idxs_1(fix(walking_count_1*train_fraction)+1:fix(walking_count_1*(train_fraction+validation_fraction)));
test_walking_idxs_1 = walking_idxs_1(fix(walking_count_1*(train_fraction+validation_fraction))+1:end);

walking_count_2 = length(walking_idxs_2);
train_walking_idxs_2 = walking_idxs_2(1:fix(walking_count_2*train_fraction));
validation_walking_idxs_2 = walking_idxs_2(fix(walking_count_2*train_fraction)+1:fix(walking_count_2*(train_fraction+validation_fraction)));
test_walking_idxs_2 = walking_idxs_2(fix(walking_count_2*(train_fraction+validation_fraction))+1:end);

walking_count_3 = length(walking_idxs_3);
train_walking_idxs_3 = walking_idxs_3(1:fix(walking_count_3*train_fraction));
validation_walking_idxs_3 = walking_idxs_3(fix(walking_count_3*train_fraction)+1:fix(walking_count_3*(train_fraction+validation_fraction)));
test_walking_idxs_3 = walking_idxs_3(fix(walking_count_3*(train_fraction+validation_fraction))+1:end);

vehicle_count = length(vehicle_idxs);
train_vehicle_idxs = vehicle_idxs(1:fix(vehicle_count*train_fraction));
validation_vehicle_idxs = vehicle_idxs(fix(vehicle_count*train_fraction)+1:fix(vehicle_count*(train_fraction+validation_fraction)));
test_vehicle_idxs = vehicle_idxs(fix(vehicle_count*(train_fraction+validation_fraction))+1:end);

clutter_count = length(clutter_idxs);
train_clutter_idxs = clutter_idxs(1:fix(clutter_count*train_fraction));
validation_clutter_idxs = clutter_idxs(fix(clutter_count*train_fraction)+1:fix(clutter_count*(train_fraction+validation_fraction)));
test_clutter_idxs = clutter_idxs(fix(clutter_count*(train_fraction+validation_fraction))+1:end);

two_walking_count = length(two_walking_idxs);
%two_walking_idxs = flip(two_walking_idxs);
train_two_walking_idxs = two_walking_idxs(1:fix(two_walking_count*train_fraction));
validation_two_walking_idxs = two_walking_idxs(fix(two_walking_count*train_fraction)+1:fix(two_walking_count*(train_fraction+validation_fraction)));
test_two_walking_idxs = two_walking_idxs(fix(two_walking_count*(train_fraction+validation_fraction))+1:end);

sphere_count = length(sphere_idxs);
train_sphere_idxs = sphere_idxs(1:fix(sphere_count*train_fraction));
validation_sphere_idxs = sphere_idxs(fix(sphere_count*train_fraction)+1:fix(sphere_count*(train_fraction+validation_fraction)));
test_sphere_idxs = sphere_idxs(fix(sphere_count*(train_fraction+validation_fraction))+1:end);

training_set_idxs = [train_sphere_idxs train_two_walking_idxs train_clutter_idxs train_vehicle_idxs train_walking_idxs_3 train_walking_idxs_2 train_walking_idxs_1 train_running_idxs_2 train_running_idxs_1];
testing_set_idxs = [test_sphere_idxs test_two_walking_idxs test_clutter_idxs test_vehicle_idxs test_walking_idxs_3 test_walking_idxs_2 test_walking_idxs_1 test_running_idxs_2 test_running_idxs_1];
validation_set_idxs = [validation_sphere_idxs validation_two_walking_idxs validation_clutter_idxs validation_vehicle_idxs validation_walking_idxs_3 validation_walking_idxs_2 validation_walking_idxs_1 validation_running_idxs_2 validation_running_idxs_1];

% Train set cross validation = Train set + Validation set
training_set_idxs_cv = [train_sphere_idxs validation_sphere_idxs train_two_walking_idxs validation_two_walking_idxs train_clutter_idxs validation_clutter_idxs train_vehicle_idxs validation_vehicle_idxs validation_walking_idxs_3 train_walking_idxs_3 train_walking_idxs_2 validation_walking_idxs_2 train_walking_idxs_1 validation_walking_idxs_1 train_running_idxs_2 validation_running_idxs_2 train_running_idxs_1 validation_running_idxs_1];
testing_set_idxs_cv = [test_sphere_idxs test_two_walking_idxs test_clutter_idxs test_vehicle_idxs test_walking_idxs_3 test_walking_idxs_2 test_walking_idxs_1 test_running_idxs_2 test_running_idxs_1];



TrainDatasetStruct =  struct('Data', {}, 'Label', {});  
TestDatasetStruct =  struct('Data', {}, 'Label', {});  
ValidationDatasetStruct =  struct('Data', {}, 'Label', {});  

TrainDatasetStructCV =  struct('Data', {}, 'Label', {});

for idx=training_set_idxs
    TrainDatasetStruct(end+1).Data = HAVSDatasetStruct(idx).Data;
    TrainDatasetStruct(end).Label = HAVSDatasetStruct(idx).Label;
end

for idx=testing_set_idxs
    TestDatasetStruct(end+1).Data = HAVSDatasetStruct(idx).Data;
    TestDatasetStruct(end).Label = HAVSDatasetStruct(idx).Label;
end

for idx=validation_set_idxs
    ValidationDatasetStruct(end+1).Data = HAVSDatasetStruct(idx).Data;
    ValidationDatasetStruct(end).Label = HAVSDatasetStruct(idx).Label;
end

% Cross validation sets 
for idx=training_set_idxs_cv
    TrainDatasetStructCV(end+1).Data = HAVSDatasetStruct(idx).Data;
    TrainDatasetStructCV(end).Label = HAVSDatasetStruct(idx).Label;
end


% Saving the data to a .mat file
save('train_data', 'TrainDatasetStruct', '-v7.3');
save('test_data', 'TestDatasetStruct', '-v7.3');
save('val_data', 'ValidationDatasetStruct', '-v7.3');

% Cross validation sets 
% save('train_data_cross_val', 'TrainDatasetStructCV', '-v7.3');
% save('test_data_cross_val', 'TestDatasetStruct', '-v7.3');


