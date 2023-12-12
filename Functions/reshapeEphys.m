function ephysx = reshapeEphys(ephysx, nchannels)
    % Reshape binary vector to N channel matrix

    ephysx = reshape(ephysx, [], 1);
    ephysx = reshape(ephysx,[nchannels, round(length(ephysx)/nchannels)]);

end