pos = 12;
x = double(trk_data_struct(pos).trk_data_real) + double(1i*trk_data_struct(pos).trk_data_imag);
stft(x,Fs,'Window',window,'OverlapLength',64,'FFTLength',fft_length);
colormap jet;