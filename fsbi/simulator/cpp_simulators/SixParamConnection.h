/* 
* Copyright 2014-2018 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#ifndef SIXPARAMCONNECTION_H_
#define SIXPARAMCONNECTION_H_

#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/DuplexConnection.h"
#include "auryn/Trace.h"
#include "auryn/LinearTrace.h"

namespace auryn {


/*! \brief Implements a parameterized STDP windows with pre and post terms. Class adapted from SymmetricSTDPConnection
 *
 * This class is adapted from SymmetricSDTPConnection, which implemented a plastic connection object implementing the plasticity rule
 * used in Vogels et al. 2011 for the inhibitory plasticity.
 *
 */
class SixParamConnection : public DuplexConnection
{

public:
	AurynFloat learning_rate;
	AurynFloat alpha_val; 
	AurynFloat beta_val;
    AurynFloat gamma_val;
    AurynFloat kappa_val;


	Trace * tr_pre;
	Trace * tr_post;

	inline AurynWeight dw_pre(NeuronID post);
	inline AurynWeight dw_post(NeuronID pre);

	inline void propagate_forward();
	inline void propagate_backward();

	bool stdp_active;

	/*! Constructor to create a random sparse connection object and set up plasticity.
	 *
	 * @param source the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param weight the initial weight of all connections.
	 * @param sparseness the connection probability for the sparse random set-up of the connections.
	 * @param eta the learning rate parameter.
	 * @param alpha plasticity rule parameter: pre term
     * @param beta plasticity rule parameter: post term.
     * @param kappa plasticity rule parameter: hebbian term at a pre spike.
     * @param gamma plasticity rule parameter: hebbian term at a post spike.
	 * @param tau_pre time constant from the pre_neuron.
     * @param tau_post time constant from the post_neuron.
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	SixParamConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05,
			AurynFloat eta=1e-3, AurynFloat alpha=-0.25, AurynFloat beta=0., AurynFloat kappa=1., AurynFloat gamma=1.,
            AurynFloat tau_pre=20e-3, AurynFloat tau_post=20e-3,
			AurynWeight maxweight=10. , TransmitterType transmitter=GABA, string name="SixParamConnection");

	/*! Constructor that creates the connection directly from a wmat file.
	 *
	 * @param source the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param filename the filename of a wmat file to build he connection from
	 * @param eta the learning rate parameter.
	 * @param alpha plasticity rule parameter: pre term
     * @param beta plasticity rule parameter: post term.
     * @param kappa plasticity rule parameter: hebbian term at a pre spike.
     * @param gamma plasticity rule parameter: hebbian term at a post spike.
	 * @param tau_pre time constant from the pre_neuron.
     * @param tau_post time constant from the post_neuron.
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	SixParamConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1e-3, AurynFloat alpha=-0.25, AurynFloat beta=0., AurynFloat kappa=1., AurynFloat gamma=1.,
            AurynFloat tau_pre=20e-3, AurynFloat tau_post=20e-3,
			AurynWeight maxweight=10 , TransmitterType transmitter=GABA);

	virtual ~SixParamConnection();
	void init(AurynFloat eta, AurynFloat alpha, AurynFloat beta, AurynFloat gamma, AurynFloat kappa, AurynFloat tau_pre, AurynFloat tau_post, 
AurynWeight maxweight);
	void free();

	virtual void propagate();

};

}

#endif /*SIXPARAMCONNECTION_H_*/