
function sorted_activity = frequencySort(sniff_activity, nchannels_ephys, nsniffs, window_size)

    sorted_activity = zeros(nsniffs, window_size, nchannels_ephys);
    for ch = 1:nchannels_ephys
        
        n_peaks = zeros(nsniffs, 1);
        
        for ii = 1:nsniffs
            freq_test = smooth(sniff_activity(ii, window_size/2:end, ch), 50, 'sgolay');
            if ~isempty(freq_test)
                [pk_test, loc_test] = findpeaks(freq_test, "MinPeakProminence", .1);
                n_peaks(ii) = loc_test(1);
            end
        end
    
        % Sort the vector and get the sorting indices
        [~, sort_indices] = sort(n_peaks);
    
        % Combine unique rows and frequencies
        result = [sniff_activity(:,:,ch), sort_indices];
    
        
            
        % Sort by frequency
        sortedResult = sortrows(result, width(result));
        
        % Extract just the sorted unique rows
        %sortedResult(:, end-1) = sortedResult(:, end);
        sorted_activity(:,:,ch) = sortedResult(:, 1:end-1);
    
    end
end
