

%----------------------------------FV-SMO----------------------------------

function  [alpha, b, ii, iter, m_M] = smo_4(X, Y, C, ker, arg, m_M, alpha, Y_grad, ker_cache, i_k, ii, eps, iter)
while m_M > eps
    I_up = find(((alpha < C) & Y == 1)|((alpha > 0) & Y == -1));      len_up = length(I_up);    I_low = find(((alpha < C) & Y == -1)|((alpha > 0) & Y == 1));     len_low = length(I_low);
    A1 = Y_grad(I_up);    [m_alpha,i1] = max(A1);    i = I_up(i1);    A2 = Y_grad(I_low);    [M_alpha,i2] = min(A2);    j = I_low(i2);
    m_M = m_alpha - M_alpha;
    if (len_up > 1 && len_low > 1)
        A1(i1 )= -inf;
        [m2,i3] = max(A1);        
        k = I_up(i3);
        A2(i2) = inf;
        [M2,i4] = min(A2);        
        l = I_low(i4);
        flag = 4;
    else 
        flag = 2;
    end
    if (k==i || k==j|| l==j|| l==j)
        flag = 2;
    end 
    if flag==4 
        [ii, ker_i,ker_j,ker_k,ker_l,ker_cache,i_k] = get_ker_cache4(ker,arg,X,i,j,k,l,i_k,ker_cache,ii);
        [Lj Uj] = compute_box(C,[Y(i),Y(j)],[alpha(i),alpha(j)]);
        [Ll Ul] = compute_box(C,[Y(k),Y(l)],[alpha(k),alpha(l)]);
        a11 = ker_i(i) + ker_j(j) - 2*ker_i(j); a22 = ker_k(k) + ker_l(l) - 2*ker_k(l); a12 = ker_i(k) - ker_i(l) - ker_j(k) + ker_j(l);
        b1 = -Y_grad(i) + Y_grad(j); b2 = -Y_grad(k) + Y_grad(l);
        tmp = a11*a22 - a12^2;
        d20 = (b1*a22-b2*a12)/tmp;
        d40 = (b2*a11-b1*a12)/tmp;
        if a12>=0
            d2=max(min(min(max(d20,(b1-a12*Ul)/a11),(b1-a12*Ll)/a11),Uj),Lj);  
            d4=max(min(min(max(d40,(b2-a12*Uj)/a22),(b2-a12*Lj)/a22),Ul),Ll);  
        else
            d2=max(min(min(max(d20,(b1-a12*Ll)/a11),(b1-a12*Ul)/a11),Uj),Lj);
            d4=max(min(min(max(d40,(b2-a12*Lj)/a22),(b2-a12*Uj)/a22),Ul),Ll);
        end
        alpha(i) = alpha(i) - d2*Y(i);   
        alpha(j) = alpha(j) + d2*Y(j);
        alpha(k) = alpha(k) - d4*Y(k); 
        alpha(l) = alpha(l) + d4*Y(l);
        Y_grad = Y_grad + d2.*ker_i - d2.*ker_j + d4.*ker_k - d4.*ker_l;
    else
        [ii, ker_i,ker_j,ker_cache,i_k] = get_ker_cache2(ker,arg,X,i,j,i_k,ker_cache,ii);
        a = ker_i(i) + ker_j(j) - 2* ker_i(j);                                                                              
        [d,alpha(i) alpha(j)] = compute_2(C,a,[Y(i),Y(j)],[alpha(i),alpha(j)],[-Y(i)*m_alpha,-Y(j)*M_alpha]);             
        Y_grad = Y_grad + d.*ker_i - d.*ker_j;                                                                               
    end 
    iter = iter + 1;   
end  
i_k1 = i_k(1:ii-1);
i_sv = find(alpha(i_k1) < C & alpha(i_k1) > 0);
tmp = (alpha.*Y)'*ker_cache(:,i_sv);
b = Y(i_k1(i_sv)) - tmp';
b = mean(b);

return; 

%----------------------------------SMO-------------------------------------

function [alpha, b, ii, iter, m_M] = smo_2(X, Y, C, ker, arg, m_M, alpha, Y_grad, ker_cache, i_k, ii, eps, iter)
while m_M > eps
    I_up = find(((alpha < C) & Y == 1)|((alpha > 0) & Y == -1));    I_low = find(((alpha < C) & Y == -1)|((alpha > 0) & Y == 1));    
    A1 = Y_grad(I_up);    [m_alpha,i1] = max(A1);    i = I_up(i1);    A2 = Y_grad(I_low);    [M_alpha,i2] = min(A2);    j = I_low(i2);
    m_M = m_alpha - M_alpha;
   [ii, ker_i,ker_j,ker_cache,i_k] = get_ker_cache2(ker,arg,X,i,j,i_k,ker_cache,ii);
    a = ker_i(i) + ker_j(j) - 2*ker_i(j);    
   [d,alpha(i), alpha(j)] =  compute_2(C,a,[Y(i),Y(j)],[alpha(i),alpha(j)],[-Y(i)*Y_grad(i),-Y(j)*Y_grad(j)]);
   Y_grad = Y_grad + d.*ker_i - d.*ker_j;
   iter = iter + 1;
end    
    i_k1 = i_k(1:ii-1);
    i_sv = find(alpha(i_k1) < C & alpha(i_k1) > 0);
    tmp = (alpha.*Y)'*ker_cache(:,i_sv);
    b = Y(i_k1(i_sv)) - tmp';
b = mean(b);
return;

function  [ii, ker_i,ker_j,ker_cache,i_k] = get_ker_cache2(ker,arg,X,i,j,i_k,ker_cache,ii)
i1 = find(i_k==i);
if isempty(i1)
    ker_i = kernel(X,X(i,:),ker,arg);
    ker_cache(:,ii) = ker_i;
    i_k(ii) = i;
    ii = ii+1;
else
    ker_i = ker_cache(:,i1);
end
j1 = find(i_k==j);
if isempty(j1)
    ker_j = kernel(X,X(j,:),ker,arg);
    ker_cache(:,ii) = ker_j;
    i_k(ii) = j;
    ii = ii+1;
else
    ker_j = ker_cache(:,j1);
end
return;
%------------------------------Kernel function-----------------------------

function K = kernel(x,y,ker,arg)
switch lower(ker)
    case 'linear'
        K = x*y';
    case 'ploy'
        c = arg(1);
        d = arg(2);
        K = (x*y'+c).^d;
    case 'rbf'        
        g = arg;
        rows = size(x,1);
        cols = size(y,1);   
        tmp = zeros(rows,cols);
        for i = 1:rows
            for j = 1:cols
                tmp(i,j) = norm(x(i,:)-y(j,:));
            end
        end        
        K = exp(-g*(tmp.^2));
    case 'tanh'
        g = arg(1);
        c = arg(2);
        K = tanh(g*x*y'+c);
    case 'ploy_rbf'
        d = arg(1);
        K1 = (x*y'+1).^d;
        s = arg(2);
        rows = size(x,1);
        cols = size(y,1);   
        tmp = zeros(rows,cols);
        for i = 1:rows
            for j = 1:cols
                tmp(i,j) = norm(x(i,:)-y(j,:));
            end
        end  
        K2 = exp(-0.5*(tmp.^2)/s);
        t = arg(3);
        K = t*K1 + (1-t)*K2;
    otherwise
        K = 0;
end










