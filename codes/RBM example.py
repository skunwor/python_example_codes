# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:59:19 2020

@author: sujitk
"""

class RBM(object):
    def __init__(
        self,
        input = None,
        n_visible = 784,
        n_hidden = 500,
        w = None,
        hbias = None,
        vbias = None,
        numpy_rng = None,
        theano_rng = None
        ):
            self.n_visible = n_visible
            self.n_hidden = n_hidden
            
            if numpy_rng is None:
                numpy_rng = numpy.random.RandomState (1234)
            
            if theano_rng is None:
                theano_rng = RandomStreams(numpy_rng.randint(2**30))

            if W is None:
                initial_W = numpy.asarray(
                    numpy_rng.uniform(
                            low = -4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                            high = 4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                            size = (n_visible, n_hidden)
                            ),
                    dtype = theano.config.floatX
                )
    

                W = theano.shared(value = initial_W, name = 'W', borrow = True)

            if hbias is None:
                hbias = theano.shared(
                    value =numpy.zeros(
                        n_hidden,
                        dtype = theano.config.floatX
                        ),
                        name = 'hbias',
                        borrow = True
                )
                    
            if vbias is None:
                vbias = theano.shared(
                    value =numpy.zeros(
                        n_visible,
                        dtype = theano.config.floatX
                        ),
                        name = 'vbias',
                        borrow = True
                )
                    
            self.input = input
            if not input:
                self.input = T.matrix('input')
            
            self.W = W
            self.hbias = hbias
            self.vbias = vbias
            self.theano_rng = theano_rng
            self.params = [self.W, self.hbias, self.vbias]


    def propup(self,vis):
        pre_sigmoid_activation = T.dot(vis,self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
    
    def propup(self,hid):
        pre_sigmoid_activation = T.dot(hid,self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
    
    def sample_h_given_h(self,v0_sample):

        presigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size = h1_mean.shape,
            n=1, p = h1_mean, dtype = theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]
    
    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return[pre_sigmoid_v1, v1_mean, v1_sample,
            pre_sigmoid_h1, h1_mean, h1_sample]
            
    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_h(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_v(h1_sample)
        return[pre_sigmoid_h1, h1_mean, h1_sample,
               pre_sigmoid_v1, v1_mean, v1_sample]
    
    def free_energy (self, v_sample):
        wx_b = T.dot(v_sample,self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)),axis = 1)
        return -hidden_term - vbias_term
    
    def get_cost_updates(self, lr = 0.1, persistent = 1, k = 1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
            
        chain_start = persistent
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
                ],
                updates
            ) = theano.scan (
                self.gibbs_hvh,
                outputs_info = [None,None,None,None,None, chain_start],
                n_steps = k
            )
        chain_end = nv_samples[-1]
        
        cost = T.mean(self.free_energy(self.input)) - T.mean(
                self.free_energy(chain_end))
        gparams = T.grad(cost,self.params, consider_constant = [chain_end])
        
        for gparam, param in zip(gparams, self.params):
            updates[Param] = param - gparam * T.cast(
                lr,
                dtype = theano.config.floatX
            )
            updates = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        return monitoring_cost, updates
    def get_pseudo_likelihood_cost(self,updates):
        bit_i_idx = theano.shared(value = 0, name = 'bit_i_idx')
        xi = T.round(self.input)
        
        fe_xi = self.free_energy(xi)
        
        xi_flip = T.set_subtensor(xi[:,bit_i_idx], 1- xi[:,bit_i_idx])
        
        fe_xi_flip = self.free_energy(xi_flip)
        
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible 
        
        return cost
    
    train_rbm = theano.function(
            [index],
            cost,
            updates = updates,
            givens = {
                x: train_set_x[index*batch_size: (index+1)*batch_size]
                },
            name= 'train_rbm'
    )
    
    plotting_time = 0
    start_time = time.clock()
    
    for epoch in xrange(training_epochs):
        
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        print('Training epoch %d, cost is ' % epoch,
             numpy.mean(mean_cost))
        plotting_start = time.clock()
        
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape = (28,28),
                tile_shape = (10,10),
                tile_spacing = (1,1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)
        
    end_time = time.clock()
    
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time /6.0))
    
    plot_every = 100 
    (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            rbm.gibbs_vhv,
            outputs_info = [None,None,None,None,None, persistent_vis_chain],
            nsteps = plot_every
        )
        
    updates.update({persistent_vis_chain:vis_samples[-1]})
    sample_fn = theano.function(
            [],
            [
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates = updates,
            name = 'sample_fn'
    )
    
    image_data = numpy.zeros(
            (29* n_samples +1, 29 * n_chains -1),
            dtype = 'uint8'
    )
    for idx in xrange (n_samples):
        vis_mf, vis_sample = sample_fn()
        print ('... plotting sample ', idx)
        image_data[29* idx:29 * idx +28, :] = tile_raster_images(
                X = vis_mf,
                img_shape = (28,28),
                tile_shape = (1, n_chains),
                tile_spacing = (1,1)
        )
    image = Image.fromarray(image_data)
    image.save(samples.png)
        
    
    
    
    
        






























