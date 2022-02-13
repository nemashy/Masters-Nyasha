im_num = 20;
x  = double(trk_data_struct(im_num).trk_data_real) + double(1i* trk_data_struct(im_num).trk_data_imag)
stft(filter(filter_params, x),Fs,'Window',window,'OverlapLength',64,'FFTLength',128);
colormap jet;