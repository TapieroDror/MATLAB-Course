function varargout = sum_var2(varargin)
switch nargout
    case 1
        s=0;
        for i=1:nargin
         s=s+varargin{i};
         varargout{1}=s;
        end
    case 2
        s=0;
        w=1;
        for i=1:nargin
            s=s+varargin{i};
            w=w+varargin{i};
        end
        varargout{1}=s;
        varargout{2}=w;
    otherwise
        error('Too many output arguments.')
end
fprintf('nargout: %d\n',nargout)
