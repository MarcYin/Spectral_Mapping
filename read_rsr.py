def read_rsr(fname):
    '''
    A python function for the reading of RSR functions from text files in https://github.com/MarcYin/RSR
    --------------------------------------------------------------------------------------------------------------------
    variables
    fname: Filename of text file contain the RSRs.
    
    return
    D: A dictionary of different colletions and bands RSRs, corresponding wavelength and center wavelength of each band.
    
    --------------------------------------------------------------------------------------------------------------------
    '''
    data = []
    with open(fname, 'rb') as f:
        for _ in xrange(12): # skip header
            next(f)
        for i in f:
            data.append(i.strip())#get rid of '\n'
    cs = []
    for k,i in enumerate(data):
        if 'c' in i :
            cs.append([k,i])
    D = {}
    for _,c in enumerate(cs):
        if len(cs)>1:
            gp = cs[1][0] - cs[0][0]
            sub = data[gp*_:gp*(_+1)]
        else:
            sub = data
        k, bands, cwl, wavelength, rsr = sub[0], sub[1::4], sub[2::4], sub[3::4], sub[4::4]

        d = {}
        for i, j in enumerate(bands):
            d.update({j:{'cwl':float(cwl[i]), 'wavelength':np.array(wavelength[i].split(',')).astype(float), 'rsr':np.array(rsr[i].split(',')).astype(float)}})
        D.update({k:d})
    
    return D