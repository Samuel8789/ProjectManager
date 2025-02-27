pkg load instrument-control;
pkg load image;


% ============================ %
%   PARAMETERS & INITIAL SETUP  %
% ============================ %
opto_params=struct();
%toggles
opto_params.test_color = false; % to choose blue or green
opto_params.all_opto=false; % only if I want to test something with all opto ON.

% Screen Parameters
opto_params.width = 1280;
opto_params.height = 720;
% Experiment Parameters
opto_params.AnimalID = '3p001'; 
opto_params.DateInfo = '20250217';
arduino_name="/dev/ttyACM0";

opto_params.opto_duration=0.27; % If laser_delay in Behavioral delay increase, this value also needs to increase.
% opto_params.intensity Settings
opto_params.intensity = 255; % 175 for red, 30 for green, 5 is min power
opto_params.missed_trials=[];
opto_params.projected_sequence=[];



% ============================ %
%   BASIC SETUP  %
% ============================ %
opto_params.mask_path = '/home/pcopto/Dropbox/ProjectorExperiment/MaskImage/';
opto_params.sequence_path = '/home/pcopto/Dropbox/ProjectorExperiment/Sequence_info/';
opto_params.output_dir='/home/pcopto/Dropbox/ProjectorExperiment/ProjectorOutput/';
opto_params.mask_names = {'final_mask_fullv1.jpg', 'final_mask_gabor_inner.jpg',...
              'final_mask_gabor_middle.jpg', 'final_mask_gabor_outer.jpg',...
              'final_mask_v1_no_gabor_inner.jpg', 'final_mask_v1_no_gabor_middle.jpg',...
              'final_mask_v1_no_gabor_outer.jpg','final_mask_v1_no_gabor_binocular.jpg',...
              'final_mask_full_screen_no_gabor_inner.jpg','final_mask_full_screen_no_gabor_middle.jpg',...
              'final_mask_full_screen_no_gabor_outer.jpg','final_mask_full_screen_no_gabor_binocular.jpg'};
              


isi_color = [0 0 0];
if opto_params.test_color
    square_color = [0 opto_params.intensity 0]; % Green
else
    square_color = [0 0 opto_params.intensity]; % Blue
end

% ============================ %
%   LOAD IMAGES INTO ARRAY      %
% ============================ %

full_screen_on = ones(opto_params.height, opto_params.width) * 255; % White
full_screen_off = zeros(opto_params.height, opto_params.width);     % black
num_files = length(opto_params.mask_names);
images = cell(num_files + 2, 1);  % Extra slots for full-screen images

% Load mask images
for i = 1:num_files
    file_path = strcat(opto_params.mask_path, opto_params.AnimalID, '/', opto_params.mask_names{i});  
    images{i} = imread(file_path);
end

% Add full-screen images
images{num_files + 1} = full_screen_off; % black
images{num_files + 2} = full_screen_on;  % white
opto_params.mask_names{num_files + 1} = 'Blank'; % black
opto_params.mask_names{num_files + 2} = 'Full On'; % white 
% ============================ %
%       LOAD SEQUENCES         %
% ============================ %

load(strcat(opto_params.sequence_path, '/sequence_', opto_params.AnimalID, '_', opto_params.DateInfo, '.mat'), 'sequence');
load(strcat(opto_params.sequence_path, '/doFullMask_', opto_params.AnimalID, '_', opto_params.DateInfo, '.mat'), 'doFullMask');

% Modify sequence for masked trials
opto_params.new_sn = sequence;
display(unique(sequence));
##display(opto_params.new_sn(doFullMask == 1)); 
opto_params.new_sn(doFullMask == 1) = 0; % 0 is do full mask.

% ============================ %
%  MANAGE SEQUENCE IMAGE MAP   %
% ============================ %

equivalence_map = struct();
final_sequence={};
for i=[0,11,13,15,19]
  equivalence_map.(num2str(i))=num_files + 2; %1 % FULL SCREEN: white 
end
for i=[1:10 21]
  equivalence_map.(num2str(i))=num_files + 1; %FULL BLANK: black
end
for i=[12]
  equivalence_map.(num2str(i))=6; % MIDDLE GABOR BLANK, V1 ILLUMINATED
end
for i=[14,17]
  equivalence_map.(num2str(i))=9 %5; % INNER BLANK, V1 ILLUMINATED
end
for i=[16,18]
  equivalence_map.(num2str(i))=7; % OUTER BLANK, V1 ILLUMINATED
end
for i=[20]
  equivalence_map.(num2str(i))=8; % BINOCULAR BLANK, V1 ILLUMINATED
end
equivalence_map.('22')  = num_files + 1;  % FULL BLANK JUST IN CASE
equivalence_map.('23')  = 2;  % INNER GABOR ILLUMINATED, V1 BLANK
equivalence_map.('24')  = 3;  % MIDDLE GABOR ILLUMINATED, V1 BLANK
equivalence_map.('25')  = 4;  % OUTER GABOR ILLUMINATED, V1 BLANK

if opto_params.all_opto
  for i=1:30
##    equivalence_map.(num2str(i))=num_files + 2; % FULLSCREEN ON
    equivalence_map.(num2str(i))=2; % FULLSCREEN ON

  end
end

% ============================ %
%        LOAD ARDUINO          %
% ============================ %

%s = serial(arduino_name, 9600,0.001); %UO 0.001 removed because it messes up.
s = serial(arduino_name, 9600);
fopen(s)
srl_flush(s)
trigger = 'q';
##display(trigger);


% ============================ %
%   SETUP PSYCHTOOLBOX SCREEN  %
% ============================ %

##PsychDefaultSetup(1);
##Screen('Preference', 'SkipSyncTests', 1);
##AssertOpenGL; % Ensure OpenGL compatibility
##
##screenid = max(Screen('Screens')); % Use external screen if available
##white = WhiteIndex(screenid);
##black = BlackIndex(screenid);
##
##[win, rect] = Screen('OpenWindow', screenid, isi_color); 
##ops.flipInterval = Screen('GetFlipInterval', win);
##resolution = Screen('Resolution', screenid);
##resopto_params.width = resolution.opto_params.width;
##resopto_params.height = resolution.opto_params.height;
##topPriorityLevel = MaxPriority(win);

% ============================ %
% CREATE TEXTURES FROM IMAGES  %
% ============================ %

texture = zeros(1, size(images, 1));
for i = 1:size(images, 1)
    texture(i) = Screen('MakeTexture', win, images{i}); 
end

% ============================ %
%       MAIN LOOP              %
% ============================ %
Screen('DrawTexture', win, texture(num_files + 1), [], [], [], [], [], square_color);
Screen('Flip', win);

triggered = false;
trial_index = 1;
% Select first texture
texture_to_project = equivalence_map.(num2str(opto_params.new_sn(trial_index)));
Screen('DrawTexture', win, texture(texture_to_project), [], [], [], [], [], square_color);
opto_params.onset_times=zeros(size(opto_params.new_sn));
opto_params.offset_times=zeros(size(opto_params.new_sn));

counter = 0;
prevState = 0;  % Track previous state of countingpin
counted=0;
while true
    % Check for keyboard input to exit
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        keyPressed = KbName(keyCode);
        if iscell(keyPressed)  % Handle multiple keypresses
            keyPressed = keyPressed{1};
        end
        if strcmpi(keyPressed, 'ESCAPE')  % If ESC key is pressed, exit loop
            disp("Experiment manually stopped.");
            break;
        end
    end

    % Read sensor values
% UO: Change this to while loop
    if s.bytesavailable
          % Read data from the serial port
          read = fread(s,3);
          trigger=char(read(1));
    end
      if trigger =='C'
          % Rising edge detected, count as 1 pulse
          printf("\n\nRunning Sequence #%d\n",trial_index);
          counter = counter + 1;
          printf("Count detected! Total Real Trial: %d\n", counter);
          printf("Count detected! Total Sequence Trial: %d\n", trial_index);
          trigger='q'; 
          % If trial_index is out of sync, reset it
          if counter>1 && trial_index ~= counter
            printf("Misalignment detected! Real Trial #%d missed",trial_index);
            opto_params.missed_trials=[opto_params.missed_trials trial_index];
            printf("Reseted Sequence Trial to: %d\n", counter);
            trial_index = counter;
          
          end
      end
      
    % Trigger only when counter matches trial_index
    if trigger =='O'
        % Flip screen on TTL trigger
        printf("Triggered on, Trial # %d, Variant %d, Opto Mask %s.\n", trial_index, opto_params.new_sn(trial_index), opto_params.mask_names{1, texture_to_project});
        opto_params.onset_times(trial_index) = Screen('Flip', win);
        WaitSecs(opto_params.opto_duration);  
        opto_params.offset_times(trial_index) = Screen('Flip', win);
        printf("Triggered off, Trial # %d, Variant %d, Opto Mask %s.\n", trial_index, opto_params.new_sn(trial_index), opto_params.mask_names{1, texture_to_project});
        printf("Laser Time: %f seconds\n", opto_params.offset_times(trial_index) - opto_params.onset_times(trial_index));
        opto_params.projected_sequence=[opto_params.projected_sequence opto_params.new_sn(trial_index)];
        % Move to the next trial
        trial_index = trial_index + 1;
        % Stop if we reach the end of the sequence
        if trial_index > length(sequence)
            break;
        end
        %srl_flush(s);
        trigger='q'; %UO any random charactor not used.
        % Prepare next texture
        texture_to_project = equivalence_map.(num2str(opto_params.new_sn(trial_index)));
        Screen('DrawTexture', win, texture(texture_to_project), [], [], [], [], [], square_color);
        WaitSecs(1);  
    end
endwhile
fclose(s);
% Show final screen
Screen('DrawTexture', win, texture(num_files + 1), [], [], [], [], [], square_color);
Screen('Flip', win);

% Get the field names (which are numeric) and sort them
fields = fieldnames(equivalence_map);
fields = str2double(fields); % Convert field names from strings to numbers
sorted_fields = sort(fields); % Sort the numeric field names

% Create a new sorted structure based on the sorted field names
sorted_equivalence_map = struct();
for i = 1:length(sorted_fields)
    field = num2str(sorted_fields(i)); % Convert back to string for accessing the structure
    sorted_equivalence_map.(field) = equivalence_map.(field); % Assign the values to the new structure
end
fields = fieldnames(sorted_equivalence_map);
for i = 1:numel(fields)
    new_field = ['Var_', fields{i}]; % Prefix with 'f_'
    sorted_equivalence_map.(new_field) = sorted_equivalence_map.(fields{i});
    sorted_equivalence_map = rmfield(sorted_equivalence_map, fields{i});
end

mkdir(strcat(opto_params.output_dir, opto_params.AnimalID, '/',opto_params.DateInfo)) 
dateString = strftime("%Y-%m-%d_%H-%M-%S", localtime(time()));
filename=strcat(opto_params.output_dir, opto_params.AnimalID, '/',opto_params.DateInfo, '/','timing_data','_',dateString,'.mat');  
save(filename,'opto_params','sorted_equivalence_map');
printf("File saved")

