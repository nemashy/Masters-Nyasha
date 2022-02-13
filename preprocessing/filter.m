Fs = 714.2857;
filter_props = fdesign.notch('N,F0,BW,Ap', 6, 0, 20, 0.2, Fs);
design(filter_props);

%fvtool(filter_params);