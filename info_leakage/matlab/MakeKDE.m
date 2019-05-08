function model = MakeKDE(data, isDiscVec)

%%% data, Np*Nd
%%% isDiscVec, Np*1
% BW, Nd*Np

data = data';

[Nd, Np] = size(data);

discreteBW = 0.001;

BW = NaN(Nd, Np);

% set discrete bandwidth
if sum(isDiscVec == 1) ~= 0
    BW(:, isDiscVec == 1) = discreteBW;
end


% calculate continuous bandwidth
if sum(isDiscVec == 0) ~= 0
    continuousBW = bwHall( data(:,isDiscVec == 0), 0 );

    % continuousBW might be invalid
    if sum( isnan(continuousBW(:)) ) ~= 0
        continuousBW = bwRot( data(:,isDiscVec==0), 0 );
        continuousBW = continuousBW + (continuousBW == 0)*0.001;

    end

    % set in BW
    for i = 1:Nd
        BW(i, isDiscVec == 0) = continuousBW(i);
    end
end



if sum( isnan(BW(:)) ) ~= 0
    % BW has NaN, raise error
    isDiscVec
    error(['BandwidthError: BW vector has NaN elements!']);
end


model = kde(data, BW);



end

