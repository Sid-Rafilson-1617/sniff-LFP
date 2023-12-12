function sniff_activity = LockSniff(locs, ephysx_rx, window_size, nsniffs, begining, nchannels_ephys)

    loc_set = locs(begining:nsniffs+begining-1);

    %creating windows centered on inhalation times (pks)
    windows = cell(nsniffs, 1);
    for ii = 1:nsniffs
        window_beg = loc_set(ii) - round(window_size/2);
        window_end = loc_set(ii) + round(window_size/2);
        windows{ii} = [window_beg window_end];
    end

    % getting ephys activity
    sniff_activity = zeros(nsniffs, window_size, nchannels_ephys);

    for ii = 1:nsniffs
        for ch = 1:nchannels_ephys
            data = ephysx_rx(ch, windows{ii}(1):windows{ii}(2) - 1);

            data_mean = mean(data);
            data_std = std(data);

            zscore_data = (data - data_mean)/data_std;
            sniff_activity(ii, :, ch) = zscore_data;
        end
    end
end


    