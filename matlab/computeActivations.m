%% try pass max pooling
net=vgg16;
%% get activations for ouzel image
img=imread('/media/innereye/1T/Repos/keras-vis/examples/vggnet/images/ouzel2.jpg');
xi=floor((500-375)/2);
xi=xi:xi+375-1;
img=imresize(img(:,xi,:),[224 224]);
fc8 = activations(net,img,39);
[~,ii]=max(fc8) % ii=21, this is okay, max activation for node 21
fc7 = activations(net,img,36);
% [~,maxi]=sort(net.Layers(39).Weights(21,:));
% maxi=maxi(end-9+1:end); % maxima weights
%% compute activation for layer fc8 from fc7 + weights
%test=sum(fc7.*net.Layers(39).Weights(21,:));
fc7relu=fc7;
fc7relu(fc7<0)=0;
my_fc8=fc7relu*net.Layers(39).Weights'+net.Layers(39).Bias';
max(test-fc8) % should be small
%% pass from conv5_3 to fc_6
conv5_3 = activations(net,img,30,'OutputAs','channels');







I=deepDreamImage(net,39,21,'PyramidLevels',3,'NumIterations',30);
I=deprocessImage(I);
figure;
imshow(I);

%% Find which nodes in previous layers are important
layer=39; % fc8
layerBelow=36; % fc7
Nfilters=9;
[~,order]=sort(net.Layers(layer).Weights(21,:));
mini=order(1:Nfilters); % minima weights
maxi=order(end-Nfilters+1:end); % maxima weights
top=uint8(zeros(438,438,3,9));
bottom=top;
for filti=1:Nfilters
    I=deepDreamImage(net,layerBelow,maxi(filti),'PyramidLevels',3,'NumIterations',30,'verbose',false);
    top(:,:,:,filti)=deprocessImage(I);
    I=deepDreamImage(net,layerBelow,mini(filti),'PyramidLevels',3,'NumIterations',30,'verbose',false);
    bottom(:,:,:,filti)=deprocessImage(I);
    disp(filti)
end
figure;
subplot(1,2,1)
montage(top)
subplot(1,2,2)
montage(bottom)
%% Go further with beaks, mid right filter (node 478)
layer=36;
layerBelow=33; % 'fc6'
filter=478;
[~,order]=sort(net.Layers(layer).Weights(filter,:));
maxi=order(end-Nfilters+1:end);
top=uint8(zeros(438,438,3,9));
for filti=1:Nfilters
    I=deepDreamImage(net,layerBelow,maxi(filti),'PyramidLevels',3,'NumIterations',30,'verbose',false);
    top(:,:,:,filti)=deprocessImage(I);
    disp(filti)
end
figure;
montage(top)

%% Go further with beaks, first filter in image above (node 486)
% FIXME, how do I cross max pooling?
layer=33;
layerBelow=30; % 'conv5_3'
filter=486;
[~,order]=sort(net.Layers(layer).Weights(filter,:));
maxi=order(end-Nfilters+1:end);
top=uint8(zeros(383,383,3,9));
for filti=1:Nfilters
    I=deepDreamImage(net,layerBelow,maxi(filti),'PyramidLevels',3,'NumIterations',10,'verbose',false);
    top(:,:,:,filti)=deprocessImage(I);
    disp(filti)
end
figure;
montage(top)
   