function h2=Addsecondaxis(varargin)

%Addsecondaxis(Lim,h,axpos,vecticks)
%
%Addsecondaxis is a simple function which adds a second axis on an existing
%plot. Unlike plotxx or plotyy it does not plot a new curve
%Addsecond axis is usefull if one wants to see the same curve as a function of 2 different absissae or ordinates. For instance a time serie as a function of
%an axis in hours and asecond axis in days
%
%Addsecondaxis(Lim) uses the vector Lim to define a second horizontal axis absissae at the top of the current plot. if Lim has two elements. Lim(1) and Lim(2) are used
%for the limits of the second axis. If Lim has more than 2 elements
%min(Lim) and max(Lim) are used for the limits of the second axis
%
%Addsecondaxis(Lim,h) where h is an axis handle add the second axis horizontal axis coordinates at the top of axes(h)
%
%Addsecondaxis(Lim,h,axpos) where axpos is a string either equal to %'x' or 'y' adds either a second absissae or ordinate
%on the second axis
%
%Addsecondaxis(Lim,h,axpos,vecticks) uses the vector vectick to define the
%ticks on the second axis
%
%Examples
%t=0:1:5*24*60; %time in mn
%y=t; figure; plot(t,y);
%Addsecondaxis(t/(24*60)) %add as second x axis in days at the top;
%the previous command is equivalent to 
%figure; plot(t,y);
%Addsecondaxis([0 5])
%to fix the ticks every 1 day
%figure; plot(t,y);
%Addsecondaxis([0 5],[],[],[0:1:5])
%if on want a second ordinate axis normalised by A
%A=5;
%figure; plot(t,y);
%Addsecondaxis(y/A,[],'y')




Lim=varargin{1};
if nargin==1|isempty(varargin{2})==1;
    h=gca;
else h=varargin{2};
end

if nargin<=2|isempty(varargin{3})==1;
    axpos='x';
else
    axpos=varargin{3};
end


set(h,'box','off');
pos=get(h,'Position');
h2=axes('position',pos);

if axpos=='x'
    if length(Lim)==2
        xlim([Lim(1), Lim(2)]);
    else xlim([min(Lim),max(Lim)]);
    end
    
    if (nargin>3)
        set(h2,'position',pos,'color','none','Ytick',[],'XaxisLocation','top','Xtick',varargin{4});
    else
        set(h2,'position',pos,'color','none','Ytick',[],'XaxisLocation','top');
    end
elseif axpos=='y'
    if length(Lim)==2;
        ylim([Lim(1), Lim(2)]);
    else ylim([min(Lim),max(Lim)])
    end
    if (nargin>3)
        set(h2,'position',pos,'color','none','Xtick',[],'YaxisLocation','right','Ytick',varargin{4})
    else
        set(h2,'position',pos,'color','none','Xtick',[],'YaxisLocation','right');
    end
else disp('error axpos should ''x'' or ''y''')
end



