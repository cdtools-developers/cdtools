"""Contains various loss functions to be used for optimization

It exposes three losses, one returning the mean squared amplitude error, one
that returns the mean squared intensity error, and one that returns the
maximum likelihood metric for a system with Poisson statistics.

"""

import torch as t

__all__ = [
    'amplitude_mse',
    'AmplitudeMSENormalizer',
    'intensity_mse',
    'IntensityMSENormalizer',
    'poisson_nll',
    'SimplePoissonNLLNormalizer',
]


def amplitude_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's amplitudes

    Calculates the mean squared error between a given set of 
    measured diffraction intensities and a simulated set.


    This function calculates the mean squared error between their
    associated amplitudes. Because this is not well defined for negative
    numbers, make sure that all the intensities are >0 before using this
    loss.
    
    Note that this is actually, by defauly, a sum-squared error. In this
    case, it is intended to be used with the loss normalization 
    
    
    
    in a ptychography model. This formulation makes it easier to compare
    error calculations between reconstructions with different minibatch
    size while keeping the loss function formally equivalent to the MSE.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    This is empirically the most useful loss function for most cases

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector values
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude
    use_sum : bool
        Default is True. If set to True, actually performs the sum squared error

    Returns
    -------
    loss : torch.Tensor
        A single value for the mean amplitude mse
    """

    # I know it would be more efficient if this function took in the
    # amplitudes instead of the intensities, but I want to be consistent
    # with all the errors working off of the same inputs

    if mask is None:
        return t.sum((t.sqrt(sim_intensities) -
                      t.sqrt(intensities))**2)
    else:
        masked_intensities = intensities.masked_select(mask)
        return t.sum((t.sqrt(sim_intensities.masked_select(mask)) -
                      t.sqrt(masked_intensities))**2)


class AmplitudeMSENormalizer(object):
    """ Normalizer for the amplitude MSE loss, used with recon.optimize

    This is a normalizer designed for use with the recon.optimize function. The
    normalization is done separately from the loss, in order to make it simple to
    use different normalization strategies for different loss metrics and to make it
    easier to work with different minibatch sizes.

    This normalizer accumulates the total number of pixels across all patterns
    during the first epoch, then divides the summed loss by this count to
    convert from sum-squared error to mean-squared error.

    The normalizer is stateful: it completes its accumulation phase on the
    first epoch and then applies the same normalization factor for all
    subsequent epochs.

    Methods
    -------
    accumulate(patterns, mask=None)
        Accumulate the normalization factor (called once per minibatch).
    normalize_loss(loss)
        Apply the accumulated normalization (called once per epoch).

    """
    
    def __init__(self):
        self.first_pass_complete = False
        self.num_pix = 0
    
    def accumulate(self, patterns, mask=None):
        if not self.first_pass_complete:
            if mask is None:
                self.num_pix += patterns.numel()
            else:
                self.num_pix += patterns.masked_select(mask).numel()
    
    def normalize_loss(self, loss):
        if not self.first_pass_complete:
            self.first_pass_complete = True
        
        return loss / self.num_pix
                      
    
def intensity_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's intensities

    Calculates the summed mean squared error between a given set of 
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of diffraction intensities. This function
    calculates the mean squared error between the intensities.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude

    Returns
    -------
    loss : torch.Tensor
        A single value for the mean intensity mse

    """
    if mask is None:
        return t.sum((sim_intensities - intensities)**2) \
            / intensities.view(-1).shape[0]
    else:
        masked_intensities = intensities.masked_select(mask)
        return t.sum((sim_intensities.masked_select(mask) -
                      masked_intensities)**2) \
                      / masked_intensities.shape[0]


class IntensityMSENormalizer(object):
    """ Normalizer for the intensity MSE loss, used with recon.optimize

    This is a normalizer designed for use with the recon.optimize function. The
    normalization is done separately from the loss, in order to make it simple to
    use different normalization strategies for different loss metrics and to make it
    easier to work with different minibatch sizes.

    This normalizer accumulates the total number of pixels across all patterns
    during the first epoch, then divides the summed loss by this count to
    convert from sum-squared error to mean-squared error.

    The normalizer is stateful: it completes its accumulation phase on the
    first epoch and then applies the same normalization factor for all
    subsequent epochs.

    Methods
    -------
    accumulate(patterns, mask=None)
        Accumulate the normalization factor (called once per minibatch).
    normalize_loss(loss)
        Apply the accumulated normalization (called once per epoch).

    """
    
    def __init__(self):
        self.first_pass_complete = False
        self.num_pix = 0
    
    def accumulate(self, patterns, mask=None):
        """Accumulate pixel counts from a batch of patterns.
        
        Parameters
        ----------
        patterns : torch.Tensor
            A tensor of measured detector patterns
        mask : torch.Tensor, optional
            A mask with ones for pixels to include and zeros for pixels to
            exclude. If provided, only masked pixels are counted.
        
        """
        if not self.first_pass_complete:
            if mask is None:
                self.num_pix += patterns.numel()
            else:
                self.num_pix += patterns.masked_select(mask).numel()
    
    def normalize_loss(self, loss):
        """Convert summed loss to mean loss by dividing by pixel count.
        
        Parameters
        ----------
        loss : torch.Tensor
            The accumulated summed loss across minibatches in an epoch
        
        Returns
        -------
        normalized_loss : torch.Tensor
            The loss divided by the total number of pixels
        
        """
        if not self.first_pass_complete:
            self.first_pass_complete = True
        
        return loss / self.num_pix

    
def poisson_nll(
        intensities,
        sim_intensities,
        mask=None,
        eps=1e-6,
        subtract_min=False
    ):
    """ Returns the Poisson negative log likelihood for simulated intensities

    Calculates the overall Poisson maximum likelihood metric using
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of intensities. This loss would be appropriate
    for detectors in a single-photon counting mode, with their output
    scaled to number of photons

    Note that this calculation ignores the log(intensities!) term in the
    full expression for Poisson negative log likelihood. This term doesn't
    change the calculated gradients so isn't worth taking the time to compute

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    The default value of eps is 1e-6 - a nonzero value here helps avoid
    divergence of the log function near zero.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude
    eps : float
        Optional, a small number to add to the simulated intensities
    subtract_min : bool
        Default is False, whether to subtract a min to produce a nonnegative output
    
    Returns
    -------
    loss : torch.Tensor
        A single value for the poisson negative log likelihood

    """
    if mask is None:
        nll = t.sum(sim_intensities+eps -
                    t.xlogy(intensities,sim_intensities+eps))
        if subtract_min:
            nll -= t.sum(intensities - t.xlogy(intensities,intensities))
            
        return nll
    else:
        masked_intensities = intensities.masked_select(mask)
        masked_sims = sim_intensities.masked_select(mask)

        nll = t.sum(masked_sims + eps - \
                    t.xlogy(masked_intensities, masked_sims+eps))
        
        if subtract_min:
            nll -= t.nansum(masked_intensities - \
                    t.xlogy(masked_intensities, masked_intensities))

        return nll


class SimplePoissonNLLNormalizer(object):
    """ Normalizer for the intensity MSE loss, used with recon.optimize

    This is a normalizer designed for use with the recon.optimize function. The
    normalization is done separately from the loss, in order to make it simple to
    use different normalization strategies for different loss metrics and to make it
    easier to work with different minibatch sizes.

    This normalizer converts raw Poisson negative log likelihood values into
    a statistic that is more interpretable for comparing reconstructions. It
    performs two operations:

    1. **Offset subtraction**: Subtracts the NLL calculated when comparing
       measured patterns to themselves (i.e., poisson_nll(data, data)). This
       represents the best-case scenario and makes the loss non-negative.

    2. **Normalization scaling**: Divides by 0.5 times the count of non-zero
       pixels in the measured patterns. This is because, roughly, each non-zero
       pixel is expected to contribute to the Poisson NLL, if Poisson noise were
       the only relevant source of noise in the data.

    The normalizer is stateful: it completes its accumulation phase on the
    first epoch by processing all patterns in the data, then applies the
    same normalization factors for all subsequent epochs.

    Methods
    -------
    accumulate(patterns, mask=None)
        Accumulate the normalization factor (called once per minibatch).
    normalize_loss(loss)
        Apply the accumulated normalization (called once per epoch).

    """
    
    def __init__(self):
        self.first_pass_complete = False
        self.sum_nonzero = 0
        self.offset = 0
    
    def accumulate(self, patterns, mask=None):
        """Accumulate statistics needed for normalization from a batch.
        
        During the first epoch, this method counts non-zero pixels and
        computes the Poisson NLL comparing patterns to themselves, which
        defines the offset baseline for the loss.
        
        Parameters
        ----------
        patterns : torch.Tensor
            A tensor of measured detector patterns
        mask : torch.Tensor, optional
            A mask with ones for pixels to include and zeros for pixels to
            exclude. If provided, only masked pixels are counted.
        
        """
        if not self.first_pass_complete:
            if mask is None:
                self.sum_nonzero += t.sum(patterns >= 1)
                self.offset += poisson_nll(patterns, patterns)
            else:
                masked_pats = patterns.masked_select(mask)
                self.sum_nonzero += t.sum(masked_pats >= 1)
                self.offset += poisson_nll(masked_pats, masked_pats)

    
    def normalize_loss(self, loss):
        """Normalize the Poisson NLL for interpretability across datasets.
        
        Parameters
        ----------
        loss : torch.Tensor
            The accumulated Poisson NLL across minibatches in an epoch
        
        Returns
        -------
        normalized_loss : torch.Tensor
            The offset-corrected and scaled loss value
        
        """
        if not self.first_pass_complete:
            self.normalization = 0.5 * self.sum_nonzero
            self.first_pass_complete = True
        
        return (loss - self.offset) / self.normalization
    
    
#
# Note: I have two other ideas for how to normalize the Poisson NLL
#
# Idea 2: Use the mean pattern to estimate the expected error
# Idea 3: Use the simulated intensities to estimate it, but use detach
#         so it doesn't hit the backward pass
#

def poisson_plus_fixed_nll(
        intensities,
        sim_intensities,
        fixed_nll,
        range,
        mask=None,
        eps=1e-6,
        subtract_min=False):
    """ Return a combined negative log likelihood for Poisson and fixed noise.

    This is unpublished as far as I know. First, it calculates the log
    likelihoods for all the possible true photon counts within (range)
    of the measured value, using the detector noise model.

    Next, it calculates the poisson log likelihood for each of those
    possible true photon counts within the minimum and maximum offset
    defined by range. It then compares them to the simulated poisson
    reciprocal arrival rate (simulated intensity). Finally, it combines
    them using a logsumexp to calculate the negative log likelihood for
    the combined noise model.
     
    This loss is appropriate for detectors with significant fixed readout
    noise which is independent of signal strength, after conversion from
    detector native units to units of photons. In the simple case,
    a gaussian nll can be passed to the fixed_nll function, but if the detector
    noise has been well characterized, 

    Note that this calculation ignores the log(intensities!) term in the
    full expression for Poisson negative log likelihood. This term doesn't
    change the calculated gradients so isn't worth taking the time to compute

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    The default value of eps is 1e-6 - a nonzero value here helps avoid
    divergence of the log function near zero.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    fixed_nll : function
        A function which calculates the fixed negative log likelihood part
    rang : tuple
        A pair (min, max) defining the relative search range for photon counts
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude
    eps : float
        Optional, a small number to add to the simulated intensities
    
    Returns
    -------
    loss : torch.Tensor
        A single value for the poisson negative log likelihood

    """
    raise NotImplementedError()
    if mask is None:
        nll = t.sum(sim_intensities+eps -
                    t.xlogy(intensities,sim_intensities+eps)) \
                    / intensities.view(-1).shape[0]
        if subtract_min:
            nll -= t.sum(intensities - t.xlogy(intensities,intensities))
            
        return nll
    else:
        masked_intensities = intensities.masked_select(mask)
        masked_sims = sim_intensities.masked_select(mask)

        nll = t.sum(masked_sims + eps - \
                    t.xlogy(masked_intensities, masked_sims+eps)) \
                    / masked_intensities.shape[0]
        
        if subtract_min:
            nll -= t.nansum(masked_intensities - \
                    t.xlogy(masked_intensities, masked_intensities)) \
                    / masked_intensities.shape[0]

        return nll
