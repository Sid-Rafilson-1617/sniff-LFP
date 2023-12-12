function ephysx_rs = resampleEphys(ephysx, nchannels)
    % Resample ephys to 1khz

    resample_factor = 30;
    N_rs = width(ephysx) / 30;
    ephysx_rs = zeros(nchannels, N_rs);

    for i = 1:nchannels
        ephysx_rs(i,:) = resample(ephysx(i,:), 1, resample_factor);
    end

end