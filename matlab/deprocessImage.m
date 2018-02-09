function x=deprocessImage(x)
% normalize tensor: center on 0., ensure std is 0.1
x = x-mean(x);
x = x./(std(x) + 1e-5);
x = x*0.1;
% clip to [0, 1]
x = x+0.5;
x(x>1)=1;
x(x<0)=0; % np.clip(x, 0, 1)
% convert to RGB array
x = uint8(x.*255);