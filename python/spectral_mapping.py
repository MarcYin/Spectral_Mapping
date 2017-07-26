import numpy as np
from scipy.interpolate import interp1d

def integrate(spectrum, bandpass, minimum, maximum, dlambda):
    '''
    do the integration of spectral curve
    args:
        spectrum - a dictionary contains wavelength and reflectance values
        bandpass - a dictionary contains wavelength and relative spectral response function
        minimum - the minimum of interpolated wavelength range
        maximum - the maximum of interpolated wavelength range
        dlambda - the interval of interpolated wavelength 

    return:
        norm - integrated reflectance
    '''
    r = interp1d(spectrum['wavelength'], spectrum['reflectance'], bounds_error=False, fill_value=0)
    b = interp1d(bandpass['wavelength'], bandpass['rsr'], bounds_error=False, fill_value=0)
    
    d = np.arange(minimum, maximum, dlambda)
    integral = np.sum(r(d) * b(d) * dlambda)
    bsum = np.sum(b(d) * dlambda)
    norm = integral / bsum
    return norm

def get_int_refs(sensor, swl = None, specs = None):
    '''
    args:
        sensor - contains the relative spectral response function of sensors
        swl - the wavelength of spectral measurements
        specs - spectral values
    
    return:
        a dictionary contain the spectral curve concolved with the bandpass, basically reflectances..
    
    '''
    bands = sensor['c1'].keys()
    b_nums = np.array([int(i[4:]) for i in bands])
    sort = np.argsort(b_nums)
    inte_refs = []
    cwls = []
    v_band = [] # some bands are out of 300-2500
    for i in np.array(bands)[sort]:
        wl = sensor['c1'][i]['wavelength']
        if min(wl)>2500:
            pass
        else:
            cwl = sensor['c1'][i]['cwl']
            cwls.append(cwl)
            rsr = sensor['c1'][i]['rsr']
            mask = np.isnan(rsr)|np.isnan(wl)
            wl, rsr = wl[~mask], rsr[~mask]
            bandpass = {'wavelength': wl, 'rsr': rsr}
            int_refs = []
            for spec in specs:
                spectrum = {'wavelength': swl, 'reflectance': spec}
                int_ref = integrate(spectrum, bandpass, 300, 2500, 1.0)
                int_refs.append(int_ref)
            inte_refs.append(int_refs)
            v_band.append(i)

    return {'int_refs': np.array(inte_refs), 'cwls':np.array(cwls), 'bands':v_band}

def fit(X, y):
    '''
    a function doing the least square fitting of X and y matrix
    
    return:
        beta_hat - used for the mapping from X to y
        var_y_given_X - variance or the standard deviation of sum of least square residual
    '''
    # solve normal equations
    beta_hat = (X.T * X).I * X.T * y
    
    # estimate y given X
    y_hat = X * beta_hat
    
    # compute the variance of y given X, var(y|X) = var(epsilon|X)
    e = y - y_hat
    var_y_given_X = np.std(e)**2 # this value is the sum (product?) of the
    # conditional variance and the variance that results from not knowing
    # the true values of beta
    return beta_hat, var_y_given_X

def evaluate(Xnew, X, beta_hat, var_y_given_X):
    '''
    use beta_hat and Xnew to predict y
    and compute the new variance of y_predict
    based on the original variance and the difficiential of Xnew to X
    
    return:
        
        y_pred - predicted y given Xnew
        var_pred - the variance of y_pred
    '''
    
    # evaluate the model at values in Xnew
    y_pred = Xnew * beta_hat
    
    # compute variance of 
    u = var_y_given_X * (1.0 + Xnew * (X.T * X).I * Xnew.T)
    var_pred = np.diag(u)
    
    return y_pred, var_pred

# copied from: http://adorio-research.org/wordpress/?p=1932
def AIC(RSS, k, n):
    """
    Computes the Akaike Information Criterion.
 
       RSS-residual sum of squares of the fitting errors.
       k  - number of fitted parameters.
       n  - number of observations.
    """
    AIC = 2 * k + n * (np.log(2 * np.pi * RSS/n) + 1)
    return AIC



def mapping(integrated_spectra, filename=None, test = True):
    '''
    args:
        integradted_spectra - should be a dictionary contain at least 2 sensors integrated reflectance value
        filename - the filename you want to store the mapping results, if None, a defult file named 
        as sepctral_mapping.txt will be used.

    return:
        mappings - a dictionary contain all of the parameters needed for the spectral mapping
    '''
    # headers of the saved file
    header = '''SensorA = target
SensorB = available
{bandsB; cwlsB; beta_hat} = bands, centre wavelengths, and coefficients in comma-delimited form
*) band numbers start counting at zero
**) centre wavelengths are expressed in nm
***) first coefficient listed is the constant

sensorA;bandA;cwlA;sensorB;{bandsB};{cwlsB};{beta_hat};AIC;var_y_given_X


'''
    
    if filename != None:
        text_file = open(filename, 'wb')
    else:
        text_file = open('spectral_mapping.txt', 'wb')# no filename specified then create one txt file
    text_file.write(header)
    
    sensors = integrated_spectra.keys()

    if len(sensors)<2:
        print 'At least 2 sensors are needed for spectral mapping!'
        raise ValueError
    else:
        mappings = {}
        
        # do some test or not...
        num = integrated_spectra[sensors[0]]['int_refs'].shape[1]
        
        if test:
            test_size = 20
            training_size = num - 20
            spectrum_numbers = np.arange(num)
            training_idxs = np.sort( np.random.choice(num, size=training_size, replace=False ) ).astype(int)
            test_idxs = np.delete(spectrum_numbers, training_idxs)
        else:
            training_idxs = np.arange(num)
       
        for sensorA in sensors:
            print 'Doing '+ sensorA+'...'
            cwlsA = integrated_spectra[sensorA]['cwls']
            band_nameA = integrated_spectra[sensorA]['bands']
            mappings[sensorA] = {}
            for bandA, cwlA in enumerate(cwlsA):
                # get observations y that we want to map to
                bA_num = int(band_nameA[bandA][4:])
                integreflA = integrated_spectra[sensorA]['int_refs'].T
                y = np.matrix( integreflA[training_idxs,:][:,bandA] ).T
                mappings[sensorA][bA_num] = {}
                mappings[sensorA][bA_num]['cwl']=cwlA
                
                for sensorB in sensors:
                    if sensorA == sensorB:
                        continue
                    mappings[sensorA][bA_num][sensorB] = {}
                    # compute distance between band of sensorA and bands of sensorB
                    cwlsB = integrated_spectra[sensorB]['cwls']
                    dist = cwlsB - cwlA
                    e = np.argsort(dist)
                    # build design matrices X with band observation of sensorB
                    integreflB = integrated_spectra[sensorB]['int_refs'].T
                    aic_previous = np.inf
                    for i in range(1, len(e)+1): 
                        bandsB = e[0:i]
                        k = len(bandsB) # number of fitted parameters
                        assert np.shape(integreflB)[0] == np.shape(integreflA)[0]
                        X_ = integreflB[training_idxs,:][:,bandsB] # of shape number of training samples x number 
                                                                    #of bands considered in linear regression
                        X = np.matrix( np.hstack(( np.ones(training_size).reshape((training_size, 1)), X_ )) )
                        # should we apply some weighting? and how?
                        beta_hat, var_y_given_X = fit(X, y)
                        y_pred, var_pred = evaluate(X, X, beta_hat, var_y_given_X)
                        RSS = (y_pred - y).T * (y_pred - y)
                        aic = float(AIC(RSS, k, num))
                        if aic < aic_previous:
                            accepted=[]
                            mappings[sensorA][bA_num][sensorB] = {'bandsB':bandsB, 'cwls':cwlsB[bandsB], \
                                                           'beta_hat':beta_hat, 'AIC':aic, 'var_y_given_X':\
                                                                  var_y_given_X}
                            aic_previous = aic
                            accepted = [bandsB, cwlsB[bandsB], beta_hat, aic, var_y_given_X]# get the accepted bands

                
                    line= '%s;%i;%f;%s;%s;%s;%s;%f;%f\n'%(sensorA, bA_num,cwlA,sensorB,\
                                                          ", ".join(map(str, accepted[0])),\
                                                          ", ".join(map(str, accepted[1])), \
                                                          ", ".join(map(str, np.array(accepted[2]).ravel())),\
                                                          accepted[3],accepted[4])
                    text_file.write(line)
    text_file.close()
    
    return mappings, training_idxs



def read_spectral_mapping(filename):
    
    '''
    A function to read in the spectral mapping txt file produced with the mapping function above
    
    args:
        
        filename
    return:
    
        mappin: spectral mappings containing all of sensors
    
    '''  
    
    data = []
    with open(filename, 'rb') as f:
        for _ in xrange(10):
            next(f)
        for i in f:
                data.append(i.strip())#get rid of '\n'   

    dat = np.array([i.split(';') for i in data])
    mappin = {}
    sensorAs =  np.unique(dat[:,0])
    for sensorA in sensorAs:
        ind = np.where(dat[:,0]== sensorA)[0]
        mappin[sensorA] = {}
        bandAs = np.unique(dat[ind][:,1]).astype(int)
        #cwlAs = np.unique(dat[ind][:,2]).astype(float)

        for bandA in bandAs:
            mappin[sensorA][bandA] = {}
            ind2 = np.where(dat[ind][:,1].astype(int)==bandA)[0]
            cwlA = float(np.unique(dat[ind][ind2,2])[0])
            mappin[sensorA][bandA]['cwl'] = cwlA
            sensorBs = dat[ind][ind2][:,3]
            for i in range(len(dat[ind][ind2][:,3])):
                sensorB, bandsB, cwlsB, beta_hat, AIC, var_y_given_X = dat[ind][ind2][i,3:]
                mappin[sensorA][bandA][sensorB] = {}
                mappin[sensorA][bandA][sensorB]['bandsB'] = np.array(bandsB.split(',')).astype(int)
                mappin[sensorA][bandA][sensorB]['cwls'] = np.array(cwlsB.split(',')).astype(float)
                mappin[sensorA][bandA][sensorB]['beta_hat'] = np.matrix(beta_hat).reshape(-1,1)
                mappin[sensorA][bandA][sensorB]['AIC'] = float(AIC)
                mappin[sensorA][bandA][sensorB]['var_y_given_X'] = float(var_y_given_X)

    return mappin
