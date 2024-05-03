def optimsharpe(data,inf,sup):
    mu = np.array([np.mean(data)]).T
    cov = np.cov(data, rowvar=0)
    def ratio(weight, mu, cov):
        Woptim = np.array(weight/sum(weight)).T
        rp = Woptim.T@mu
        vp = (np.sqrt(Woptim.T@cov@Woptim))
        ratio = (rp/vp)
        return (-ratio)
    optim = minimize(fun=ratio,
                 args=(mu, cov),
                 x0=np.repeat(1/len(mu), len(mu)),
                 method= 'L-BFGS-B',
                 bounds =((inf,sup),)*len(mu),
                 constraints=({'type': 'eq', 'fun' : lambda Woptim: np.sum(Woptim)-1}))
    Wsharpe = np.array([optim.x/sum(optim.x)]).T
    Return = Wsharpe.T@mu
    Risk = np.sqrt(Wsharpe.T@cov@Wsharpe)
    sharperatio = Return/Risk
    Rpsharpe = []
    for i in data.index:
        Rp = Wsharpe.T@data.loc[i]
        Rpsharpe.append(Rp)
    return['Return',Return,'Risk',Risk,'Sharpe Ratio',sharperatio, 'Weight Sharpe', Wsharpe, Rpsharpe]


