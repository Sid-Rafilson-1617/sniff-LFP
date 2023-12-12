function [inhalation_times_seconds, sniff_frequencies] = sniff_process(sniff, f, visualize)

    % smooth signal and asign time in seconds
    degree = 25;
    sniff = smooth(sniff, degree,'sgolay');
    times = (1:length(sniff))./f;

    min_distance = f/30;
    min_prominance = 0.5;

    %peak detection 
    fprintf('begining peak detection\n')
    inhale_times = [];
    exhale_times = [];
    if ~isempty(sniff)
        windows = round(1:(5*f):length(sniff));
        zniff = zeros(1, length(windows));
        for scan = 2:length(windows)

            time_stamp=(windows(scan-1)):windows(scan);
            zniff=zscore(sniff(time_stamp));
            
            % peak prominance
            [sniff_pks,in_locs] = findpeaks(zniff,'MinPeakDistance',min_distance,'MinPeakProminence',min_prominance);
            [ex_pks,ex_locs] = findpeaks(-zniff,'MinPeakDistance',min_distance,'MinPeakProminence',min_prominance);
                
        
                %visual validation of peak finder
                if (scan == 5) && (visualize == true)
                    figure;
                    hold on
                    plot(zniff)
                    plot(sniff(time_stamp), "Color" , "Black")
                    plot(in_locs,zniff(in_locs),'ro')
                    plot(ex_locs,zniff(ex_locs),'go')
                    title('Peak Finder Validation')
                    hold off
                end

                for peak= 1:length(in_locs)
                    inhale=in_locs(peak);
                    exhale=ex_locs(ex_locs>inhale);
                    if ~isempty(exhale)
                        inhale_times=[inhale_times; inhale + windows(scan-1)];
                        exhale_times=[exhale_times; exhale + windows(scan-1)];
                        sniff_frequencies = f./diff(inhale_times);
                        inhalation_times_seconds = inhale_times(2:end)./f;                       
                    end
                end
        end
        fprintf('Peak detection complete\n\n');
    end
end