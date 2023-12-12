function sniff = removeJumps(sniff)

    % correct jumps in sniff signal
    for x = 1:length(sniff)
        if sniff(x) > 40000
            sniff(x) = sniff(x) - 65520;
        end
    end

    % correct jumps in ephys signal
%     for ch = 1:nchannels
%         for x = 1:length(ephysx(ch,:))
%             if ephysx(ch, x) > 40000
%                 ephysx(ch, x) = ephysx(ch, x) - 65520;
%             end
%         end
%     end
% 
% end