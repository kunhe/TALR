function weights = init_weights(k,m,n)
weights{1} = randn(k,k,m,n,'single') * sqrt(2/(k*k*m)) ;
weights{2} = zeros(n,1,'single') ;
end
