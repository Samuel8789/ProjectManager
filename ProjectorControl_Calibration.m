pkg load image; % For image display
pkg load instrument-control;

function update_gui_display(ax, image_data, intensity)
  % Update GUI plot with the current image
  image(ax, image_data * intensity / 255); % Scale intensity
  colormap(ax, gray); % Grayscale colormap
  axis(ax, 'off'); % Hide axis
  drawnow;
endfunction

function update_intensity(hObject, ~, win, texture1, texture2, ax, label)
  % Get the slider value
  intensity = round(get(hObject, 'Value'));

  % Update the label with the current intensity value
  set(label, 'String', sprintf('Intensity: %d', intensity));

  % Get the ToggleButton handle
  btn = findobj('Tag', 'ToggleButton');

  % Ensure UserData is a structure
  if ~isstruct(get(btn, 'UserData'))
    set(btn, 'UserData', struct('state', false, 'intensity', intensity));
  else
    data = get(btn, 'UserData');
    data.intensity = intensity;
    set(btn, 'UserData', data);
  endif

  % Get the current screen state
  screen_state = get(btn, 'UserData').state;

  % Update Psychtoolbox display first
  ProjectedColor = [0 intensity 0]; % Green intensity

  if screen_state
    Screen('DrawTexture', win, texture1, [], [], [], [], [], ProjectedColor);
  else
    Screen('DrawTexture', win, texture2, [], [], [], [], [], ProjectedColor);
  end
  Screen('Flip', win);

  % Update the image display
  image_data = ones(720, 1280) * (screen_state * 255);
  update_gui_display(ax, image_data, intensity);
endfunction

function toggle_screen(hObject, ~, win, texture1, texture2, ax)
  % Ensure UserData is a structure
  if ~isstruct(get(hObject, 'UserData'))
    set(hObject, 'UserData', struct('state', false, 'intensity', 0));
  endif

  data = get(hObject, 'UserData');
  data.state = ~data.state;
  set(hObject, 'UserData', data);

  % Update Psychtoolbox display first
  ProjectedColor = [0 data.intensity 0]; % Green intensity
  %ProjectedColor = [0 0 data.intensity]; % Green intensity
  if data.state
    Screen('DrawTexture', win, texture1, [], [], [], [], [], ProjectedColor);
  else
    Screen('DrawTexture', win, texture2, [], [], [], [], [], ProjectedColor);
  end
  Screen('Flip', win);

  % Update the image display
  image_data = ones(720, 1280) * (data.state * 255);
  update_gui_display(ax, image_data, data.intensity);
endfunction

function update_grid(~, ~, win, ax, square_size_input, spacing_input, intensity_input, rows_input, rows_intensity_input)
  % Get grid parameters from input fields
  square_size = str2double(get(square_size_input, 'String'));
  spacing = str2double(get(spacing_input, 'String'));
  default_intensity = str2double(get(intensity_input, 'String'));
  num_rows_to_change = str2double(get(rows_input, 'String'));
  rows_intensity = str2double(get(rows_intensity_input, 'String'));

  rows = 720;
  cols = 1280;
  fullgrid = zeros(rows, cols);

  % Calculate the number of squares that fit in the grid
  num_squares_x = floor((cols - spacing) / (square_size + spacing));
  num_squares_y = floor((rows - spacing) / (square_size + spacing));

  % Generate the grid with default intensity
  for i = 1:num_squares_x
      for j = 1:num_squares_y
          x_start = spacing + (i - 1) * (square_size + spacing);
          y_start = spacing + (j - 1) * (square_size + spacing);
          x_end = x_start + square_size - 1;
          y_end = y_start + square_size - 1;

          % Ensure the square does not exceed the image boundaries
          if x_end <= cols && y_end <= rows
              fullgrid(y_start:y_end, x_start:x_end) = default_intensity;
          endif
      end
  end

  % Apply custom intensity to the squares in the selected rows (top rows)
  if num_rows_to_change > 0
      for j = 1:num_rows_to_change
          y_start = spacing + (j - 1) * (square_size + spacing);
          y_end = y_start + square_size - 1;

          % Ensure the row does not exceed the image boundaries
          if y_end <= rows
              for i = 1:num_squares_x
                  x_start = spacing + (i - 1) * (square_size + spacing);
                  x_end = x_start + square_size - 1;

                  % Apply custom intensity only to the squares in the selected rows
                  fullgrid(y_start:y_end, x_start:x_end) = rows_intensity;
              end
          endif
      end
  endif

  % Display the grid in the GUI
  image(ax, fullgrid);
  colormap(ax, gray);
  axis(ax, 'off');
  drawnow;
##  ProjectedColor = [0 default_intensity 0]; % Green intensity
  ProjectedColor = [0 0 default_intensity]; % Green intensity

  % Update Psychtoolbox display with the grid
  grid_texture = Screen('MakeTexture', win, fullgrid);
  Screen('DrawTexture', win, grid_texture,[], [], [], [], [], ProjectedColor);
  Screen('Flip', win);
endfunction

function close_gui(src, ~, win)
  try
    % Close the Psychtoolbox window
    if exist('win', 'var') && ~isempty(win)
      Screen('CloseAll');
    endif

    % Delete the GUI figure
    if ishandle(src)
      delete(src);
    endif
  catch ME
    % Display any errors that occur during cleanup
    disp('Error during cleanup:');
    disp(ME.message);
  end_try_catch
endfunction

function save_grid_as_mat(~, ~, ax)
  % Save the current grid image as a .mat file
  % Retrieve the image data from the axes' CData property
  h = findobj(ax, 'Type', 'image'); % Find the image object in the axes
  if isempty(h)
    disp('No image found in the axes.');
    return;
  endif
  grid_data = get(h, 'CData'); % Get the image data
  save('grid_image.mat', 'grid_data'); % Save as .mat file
  disp('Grid saved as grid_image.mat');
endfunction

% Callback for the color toggle button
function toggle_color_button(hObject, ~)
  % Toggle between green and blue
  data = get(hObject, 'UserData');
  if data.state == 1
    set(hObject, 'String', 'Blue');
    set(hObject, 'BackgroundColor', [0 0 1]);
    data.state = 0;
  else
    set(hObject, 'String', 'Green');
    set(hObject, 'BackgroundColor', [0 1 0]);
    data.state = 1;
  endif
  set(hObject, 'UserData', data);
endfunction

function toggle_all_opto_button(hObject, ~)
  % Toggle between normal trial structure and all trial opto
  data = get(hObject, 'UserData');
  if data.state == 1
    set(hObject, 'String', 'All Trial Opto');
    set(hObject, 'BackgroundColor', [1 0 0]);
    data.state = 0;
  else
    set(hObject, 'String', 'Normal Trial Structure');
    set(hObject, 'BackgroundColor', [1 1 1]);
    data.state = 1;
  endif
  set(hObject, 'UserData', data);
endfunction
function run_full_experiment(~, ~)
    disp('Temporary PLaceholder');

endfunction
function create_gui()
  try
    % GUI stays on primary screen
    gui_screenid = min(Screen('Screens'));
    fig = figure('Position', [50, 50, 1000, 1000], 'MenuBar', 'none', 'Name', 'Screen Control', 'NumberTitle', 'off'); % Increased height to accommodate new elements

    % Psychtoolbox Setup (Fullscreen on secondary screen)
    PsychDefaultSetup(1);
    Screen('Preference', 'SkipSyncTests', 1);
    AssertOpenGL;

    exp_screenid = max(Screen('Screens'));
    [win, rect] = Screen('OpenWindow', exp_screenid, [0 0 0]); % Fullscreen mode

    % Image Setup
    width = 1280;
    height = 720;
    image1 = ones(height, width) * 255; % White
    image2 = ones(height, width) * 0;   % Black
    texture1 = Screen('MakeTexture', win, image1);
    texture2 = Screen('MakeTexture', win, image2);

    % Initial intensity set to 0
    intensity = 0;
    ProjectedColor = [0 intensity 0]; 

    % Start with black screen
    Screen('DrawTexture', win, texture2, [], [], [], [], [], ProjectedColor);

    Screen('Flip', win);

    % Axes for live image plot (normalized position)
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.1, 0.6, 0.8, 0.39]);
    update_gui_display(ax, image1, intensity);

    % Toggle button (normalized position)
    btn = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Toggle Screen', ...
                    'Units', 'normalized', 'Position', [0.1, 0.54, 0.3, 0.05], ...
                    'Callback', {@toggle_screen, win, texture1, texture2, ax}, ...
                    'Tag', 'ToggleButton');
    set(btn, 'UserData', struct('state', 0, 'intensity', intensity));

    % Intensity label (normalized position)
    label = uicontrol(fig, 'Style', 'text', ...
                      'Units', 'normalized', 'Position',[0.1, 0.44, 0.3, 0.05], ...
                      'String', sprintf('Intensity: %d', intensity), ...
                      'FontSize', 12);

    % Intensity slider (normalized position)
    slider = uicontrol(fig, 'Style', 'slider', 'Min', 0, 'Max', 255, 'Value', intensity, ...
                       'Units', 'normalized', 'Position', [0.1, 0.49, 0.3, 0.05], ...
                       'Callback', {@update_intensity, win, texture1, texture2, ax, label});

    % Input fields for grid parameters
    uicontrol(fig, 'Style', 'text', 'String', 'Square Size:', 'Units', 'normalized', 'Position', [0.5, 0.54, 0.2, 0.05]);
    square_size_input = uicontrol(fig, 'Style', 'edit', 'String', '20', 'Units', 'normalized', 'Position', [0.7, 0.54, 0.2, 0.05]);

    uicontrol(fig, 'Style', 'text', 'String', 'Spacing:', 'Units', 'normalized', 'Position', [0.5, 0.49, 0.2, 0.05]);
    spacing_input = uicontrol(fig, 'Style', 'edit', 'String', '60', 'Units', 'normalized', 'Position', [0.7, 0.49, 0.2, 0.05]);

    uicontrol(fig, 'Style', 'text', 'String', 'Intensity:', 'Units', 'normalized', 'Position', [0.5, 0.44, 0.2, 0.05]);
    intensity_input = uicontrol(fig, 'Style', 'edit', 'String', '255', 'Units', 'normalized', 'Position', [0.7, 0.44, 0.2, 0.05]);

    % New input fields for number of rows and row intensity
    uicontrol(fig, 'Style', 'text', 'String', 'Number of Rows:', 'Units', 'normalized', 'Position', [0.5, 0.39, 0.2, 0.05]);
    rows_input = uicontrol(fig, 'Style', 'edit', 'String', '0', 'Units', 'normalized', 'Position', [0.7, 0.39, 0.2, 0.05]); % Default value set to 0

    uicontrol(fig, 'Style', 'text', 'String', 'Row Intensity:', 'Units', 'normalized', 'Position', [0.5, 0.34, 0.35, 0.05]);
    rows_intensity_input = uicontrol(fig, 'Style', 'edit', 'String', '20', 'Units', 'normalized', 'Position', [0.7, 0.34, 0.2, 0.05]);

    % Save Grid as .mat button (left of Update Grid)
    save_mat_btn = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Save Grid as .mat', 'Units', 'normalized', ...
                             'Position', [0.1, 0.29, 0.3, 0.05], 'Callback', {@save_grid_as_mat, ax});
    % Grid display button moved to the right column
    grid_btn = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Update Grid', 'Units', 'normalized', ...
                         'Position', [0.5, 0.29, 0.4, 0.05], 'Callback', {@update_grid, win, ax, square_size_input, spacing_input, intensity_input, rows_input, rows_intensity_input});

    % Add a separating line below the "Update Grid" button
    uicontrol(fig, 'Style', 'text', 'String', '', 'Units', 'normalized', ...
              'Position', [0.05, 0.28, 0.95, 0.01], 'BackgroundColor', [0 0 0]);

    % New input fields below the separating line (3 columns)
    % Column 1: Mouse name and date
    uicontrol(fig, 'Style', 'text', 'String', 'Mouse Name:', 'Units', 'normalized', 'Position', [0.05, 0.23, 0.15, 0.05]);
    mouse_input = uicontrol(fig, 'Style', 'edit', 'String', '3p00', 'Units', 'normalized', 'Position', [0.2, 0.23, 0.10, 0.05]);

    uicontrol(fig, 'Style', 'text', 'String', 'Date:', 'Units', 'normalized', 'Position', [0.05, 0.18, 0.15, 0.05]);
    date_input = uicontrol(fig, 'Style', 'edit', 'String', datestr(now, 'YYYYMMDD'), 'Units', 'normalized', 'Position', [0.2, 0.18, 0.10, 0.05]);

    % Column 2: Arduino name and opto parameters
    uicontrol(fig, 'Style', 'text', 'String', 'Arduino Name:', 'Units', 'normalized', 'Position', [0.32, 0.23, 0.15, 0.05]);
    arduino_input = uicontrol(fig, 'Style', 'edit', 'String', 'ACM0', 'Units', 'normalized', 'Position', [0.47, 0.23, 0.10, 0.05]);

    uicontrol(fig, 'Style', 'text', 'String', 'Opto Duration:', 'Units', 'normalized', 'Position', [0.32, 0.18, 0.15, 0.05]);
    opto_duration_input = uicontrol(fig, 'Style', 'edit', 'String', '0.27', 'Units', 'normalized', 'Position', [0.47, 0.18, 0.10, 0.05]);

    uicontrol(fig, 'Style', 'text', 'String', 'Opto Intensity:', 'Units', 'normalized', 'Position', [0.32, 0.13, 0.15, 0.05]);
    opto_intensity_input = uicontrol(fig, 'Style', 'edit', 'String', '10', 'Units', 'normalized', 'Position', [0.47, 0.13, 0.10, 0.05]);

    % Column 3: Toggle buttons
    color_button = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Green', 'Units', 'normalized', ...
                             'Position', [0.6, 0.2, 0.11, 0.08], 'BackgroundColor', [0 1 0], ...
                             'Callback', @toggle_color_button);
    set(color_button, 'UserData', struct('state', 1)); % 1 = green, 0 = blue

    all_opto_button = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Normal Trial Structure', 'Units', 'normalized', ...
                                'Position', [0.78, 0.2, 0.2, 0.08], 'BackgroundColor', [1 1 1], ...
                                'Callback', @toggle_all_opto_button);
    set(all_opto_button, 'UserData', struct('state', 1)); % 1 = normal, 0 = all_opto

    % Add "Run Full Experiment" button
    run_experiment_btn = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Run Full Experiment', ...
                                   'Units', 'normalized', 'Position', [0.3, 0.05, 0.4, 0.05], ...
                                   'Callback', {@run_full_experiment});

    % Set the CloseRequestFcn to properly close the GUI
    set(fig, 'CloseRequestFcn', {@close_gui, win});
  catch ME
    % Display any errors that occur during GUI creation
    disp('Error during GUI creation:');
    disp(ME.message);
    % Ensure Psychtoolbox window is closed
    Screen('CloseAll');
  end_try_catch
endfunction



% Create the GUI
create_gui();
