function ephysx = LoadEphys(ephys_file)

    ephys = fopen(ephys_file, 'r'); 
    ephysx = fread(ephys, 'uint16');
    fclose(ephys);

    nchannels = 16;
    ephysx = reshape(ephysx, [], 1);
    ephysx = reshape(ephysx,[nchannels, round(length(ephysx)/nchannels)]); % Reshape binary vector to N channel matrix

end