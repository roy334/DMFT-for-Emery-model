function X = symPlaq(X)
% Enforce the C4v structure of a site-basis plaquette matrix function
% X(w,a,b) (sites F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1)):
% average over the three classes on-site / nearest neighbour / diagonal.
P = [1 2 2 3; 2 1 3 2; 2 3 1 2; 3 2 2 1];
Nw = size(X,1);
Xr = reshape(X,Nw,16);
for c = 1:3
   idx = find(P(:) == c);
   Xr(:,idx) = repmat(mean(Xr(:,idx),2),1,numel(idx));
end
X = reshape(Xr,size(X));
end
