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

#include "SixParamConnection.h"

using namespace auryn;

void SixParamConnection::init(AurynFloat eta, AurynFloat alpha, AurynFloat beta, AurynFloat gamma, AurynFloat kappa,
        AurynFloat tau_pre, AurynFloat tau_post, AurynWeight maxweight)
{
	set_max_weight(maxweight);
	set_min_weight(0.0);

	learning_rate = eta;
	alpha_val = alpha;
	beta_val = beta;
	gamma_val = gamma;
	kappa_val = kappa;


	stdp_active = true;
	if ( learning_rate == 0 )
		stdp_active = false;

	set_name("SixParamConnection");

	if ( dst->get_post_size() == 0 ) return;

	tr_pre = src->get_pre_trace(tau_pre);
	tr_post = dst->get_post_trace(tau_post);

}

void SixParamConnection::free()
{
}

SixParamConnection::SixParamConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat eta, AurynFloat alpha, AurynFloat beta, AurynFloat gamma, AurynFloat kappa,
        AurynFloat tau_pre, AurynFloat tau_post,
		AurynWeight maxweight , TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
	init(eta , alpha, beta, gamma, kappa, tau_pre, tau_post, maxweight);
}

SixParamConnection::SixParamConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, AurynFloat alpha, AurynFloat beta, AurynFloat gamma, AurynFloat kappa,
        AurynFloat tau_pre, AurynFloat tau_post,
		AurynWeight maxweight , TransmitterType transmitter, std::string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(eta , alpha, beta, gamma, kappa, tau_pre, tau_post, maxweight);
}

SixParamConnection::~SixParamConnection()
{
	free();
}

inline AurynWeight SixParamConnection::dw_pre(NeuronID post)
{
	if (stdp_active) {
		double dw = learning_rate*(alpha_val + kappa_val*tr_post->get(post));
		return dw;
	}
	else return 0.;
}

inline AurynWeight SixParamConnection::dw_post(NeuronID pre)
{
	if (stdp_active) {
		double dw = learning_rate*(beta_val + gamma_val*tr_pre->get(pre));
		return dw;
	}
	else return 0.;
}

inline void SixParamConnection::propagate_forward()
{
	NeuronID * ind = w->get_row_begin(0); // first element of index array
	AurynWeight * data = w->get_data_begin();
	AurynWeight value;
	SpikeContainer::const_iterator spikes_end = src->get_spikes()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) {
			value = data[c-ind]; 
			// dst->tadd( *c , value , transmitter );
			transmit( *c, value );
			NeuronID translated_spike = dst->global2rank(*c);
			  data[c-ind] += dw_pre(translated_spike);
			if (data[c-ind] < get_min_weight()) {
				data[c-ind] = get_min_weight();
			}
			else if (data[c-ind] > get_max_weight()) {
				data[c-ind] = get_max_weight();
			}
		}
	}
}

inline void SixParamConnection::propagate_backward()
{
	NeuronID * ind = bkw->get_row_begin(0); // first element of index array
	AurynWeight ** data = bkw->get_data_begin();
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

			#if defined(CODE_ACTIVATE_PREFETCHING_INTRINSICS) && defined(CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY)
			_mm_prefetch((const char *)data[c-ind+2],  _MM_HINT_NTA);
			#endif
			
			*data[c-ind] += dw_post(*c);
			if (*data[c-ind] < get_min_weight()) {
				*data[c-ind] = get_min_weight();
			}
			else if (*data[c-ind] > get_max_weight()) {
				*data[c-ind] = get_max_weight();
			}
		}
	}
}

void SixParamConnection::propagate()
{
	// propagate
	propagate_forward();
	propagate_backward();
}

