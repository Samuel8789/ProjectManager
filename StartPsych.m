% ============================ %
%   SETUP PSYCHTOOLBOX SCREEN  %
% ============================ %

PsychDefaultSetup(1);
Screen('Preference', 'SkipSyncTests', 1);
AssertOpenGL; % Ensure OpenGL compatibility
isi_color = [0 0 0];
screenid = max(Screen('Screens')); % Use external screen if available
white = WhiteIndex(screenid);
black = BlackIndex(screenid);

[win, rect] = Screen('OpenWindow', screenid, isi_color); 
ops.flipInterval = Screen('GetFlipInterval', win);
resolution = Screen('Resolution', screenid);
reswidth = resolution.width;
resheight = resolution.height;
topPriorityLevel = MaxPriority(win);
