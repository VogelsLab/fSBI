/* 
* This is a part of Synapseek, written by Basile Confavreux. 
* It uses the spiking network simulator Auryn written by Friedemann Zenke.
*
*                   ON PRE UPDATE
* xpre2
* xpost1            |
* w                 |            |
* <V>     --Wh1-->  |  --Wh2-->  | --Wpre--> dw*eta (linear)
* Cexc    shared    |   shared   |  
* Cinh             nh1          nh2
*                 (sig)        (sig)
* (no bias)     (no bias)      (BIAS)
*
*                   ON POST UPDATE
* xpre1
* xpost2            |
* w                 |            |
* <V>     --Wh1-->  |  --Wh2-->  | --Wpost--> dw*eta (linear)
* Cexc    shared    |   shared   |  
* Cinh             nh1          nh2
*                 (sig)        (sig)
* (no bias)     (no bias)      (BIAS)
*/

#include "T4wvceciMLPConnection.h"

using namespace auryn;

void T4wvceciMLPConnection::init(std::vector<float> coeffs, AurynWeight maxweight, int nh1_, int nh2_)
{
	set_max_weight(maxweight);
	set_min_weight(0.0);

	stdp_active = true;

	if ( dst->get_post_size() == 0 ) return;

    nh1 = nh1_;
    nh2 = nh2_;

    //rescale inputs to the MLP for more useful computation with sigmoid
    rescale_trace=2.; 
	rescale_v=15.;
	rescale_w=1.;
	rescale_cexc=15.;
	rescale_cinh=15.;

    eta = coeffs[0];

	tr_pre1 = src->get_pre_trace(coeffs[1]);
	tr_pre2 = src->get_pre_trace(coeffs[2]);
	tr_post1 = dst->get_post_trace(coeffs[3]);
	tr_post2 = dst->get_post_trace(coeffs[4]);

    n_coeffs_Wh1 = 6*nh1;
	Wh1.resize(n_coeffs_Wh1);
    for (int i=0; i<n_coeffs_Wh1; i++){
        Wh1[i] = coeffs[i+5];
    }

    n_coeffs_Wh2 = nh1*nh2;
	Wh2.resize(n_coeffs_Wh2);
    for (int i=0; i<n_coeffs_Wh2; i++){
        Wh2[i] = coeffs[i+5+n_coeffs_Wh1];
    }

    n_coeffs_Wpre = nh2+1; //the bias
    Wpre.resize(n_coeffs_Wpre); 
    for (int i=0; i<n_coeffs_Wpre; i++){
        Wpre[i] = coeffs[i+5+n_coeffs_Wh1+n_coeffs_Wh2];
    }

    n_coeffs_Wpost = nh2+1; //the bias
    Wpost.resize(n_coeffs_Wpost); 
    for (int i=0; i<n_coeffs_Wpost; i++){
        Wpost[i] = coeffs[i+5+n_coeffs_Wh1+n_coeffs_Wh2+n_coeffs_Wpre];
    }

    xh1.resize(nh1);
    for (int i=0; i<nh1; i++){
        xh1[i] = 0;
    }
    xh2.resize(nh2); 
    for (int i=0; i<nh2; i++){
        xh2[i] = 0;
    }

    // //////////////////////////////////////////////
    // std::cout << "Inside Connection: " << std::endl;
    // std::cout << "eta=" << eta << std::endl;
    // std::cout << "tau_pre1=" << tr_pre1->get_tau() << std::endl;
    // std::cout << "tau_pre2=" << tr_pre2->get_tau() << std::endl;
    // std::cout << "tau_post1=" << tr_post1->get_tau() << std::endl;
    // std::cout << "tau_post2=" << tr_post2->get_tau() << std::endl;
    // std::cout << "Wh1: " << std::endl;
    // for (int i = 0; i < n_coeffs_Wh1; i ++)
    // {
    //     std::cout << Wh1[i] << ", ";
    // }
    // std::cout << "; " << std::endl;

    // std::cout << "Wh2: " << std::endl;
    // for (int i = 0; i < n_coeffs_Wh2; i ++)
    // {
    //     std::cout << Wh2[i] << ", ";
    // }
    // std::cout << "; " << std::endl;

    // std::cout << "Wpre: " << std::endl;
    // for (int i = 0; i < n_coeffs_Wpre; i ++)
    // {
    //     std::cout << Wpre[i] << ", ";
    // }
    // std::cout << "; " << std::endl;

    // std::cout << "Wpost: " << std::endl;
    // for (int i = 0; i < n_coeffs_Wpost; i ++)
    // {
    //     std::cout << Wpost[i] << ", ";
    // }
    // std::cout << "; " << std::endl;
    // //////////////////////////////////////////////
}

void T4wvceciMLPConnection::free()
{
}

T4wvceciMLPConnection::T4wvceciMLPConnection(CVAIFGroup* source, CVAIFGroup* destination, 
			AurynWeight weight, AurynFloat sparseness, std::vector<float> coeffs, int nh1_, int nh2_,
			AurynWeight maxweight, TransmitterType transmitter, string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(coeffs, maxweight, nh1_, nh2_);
    dst_cvaif = destination; //allows to acces vtrace, cexh cinh from CVAIFGroup
}

T4wvceciMLPConnection::~T4wvceciMLPConnection()
{
	free();
}

inline AurynWeight T4wvceciMLPConnection::dw_pre(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat Cexc, AurynFloat Cinh)
{
    return eta*forward_pre(tr_pre2->get(pre), tr_post1->get(post), current_w, V, Cexc, Cinh);
}

inline AurynWeight T4wvceciMLPConnection::dw_post(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat Cexc, AurynFloat Cinh)
{
    return eta*forward_post(tr_pre1->get(pre), tr_post2->get(post), current_w, V, Cexc, Cinh);
}

inline void T4wvceciMLPConnection::propagate_forward()
{
    // loop over all spikes: spike = pre_spike
    for (SpikeContainer::const_iterator spike = src->get_spikes()->begin(); spike != src->get_spikes()->end(); ++spike) 
    {
        // loop over all postsynaptic partners (c: untranslated post index)
        for (const NeuronID* c = w->get_row_begin(*spike); c != w->get_row_end(*spike); ++c) 
        {
            // transmit signal to target at postsynaptic neuron (no plasticity yet)
            AurynWeight* weight = w->get_data_ptr(c);
            transmit(*c, *weight);
 
            // handle plasticity
            if (stdp_active) 
            {
                // translate postsynaptic spike (required for mpi run)
                NeuronID trans_post_ind = dst->global2rank(*c);

                // perform weight update
                ////////////////////////////////////////////////////////////////////////////////////////////////////
                // std::cout << " " << std::endl;
                // std::cout << "Inside propagate forward:" << std::endl;
                // std::cout << "pre index " << *spike 
                //           << ", post index " << trans_post_ind
                //           << ", voltage trace " << dst_cvaif->vtrace->get(trans_post_ind)
                //           << ", cexc " << dst_cvaif->cexc->get(trans_post_ind) 
                //           << ", cinh " << dst_cvaif->cinh->get(trans_post_ind) << std::endl;
                // std::cout << " " << std::endl;
                ////////////////////////////////////////////////////////////////////////////////////////////////////
                *weight += dw_pre(*spike, trans_post_ind, *weight, dst_cvaif->vtrace->get(trans_post_ind),
                            dst_cvaif->cexc->get(trans_post_ind), dst_cvaif->cinh->get(trans_post_ind));
               
                // clip weights if needed
                if (*weight > get_max_weight()) *weight = get_max_weight();
                if (*weight < get_min_weight()) *weight = get_min_weight();
            }
        }
    }
}

inline void T4wvceciMLPConnection::propagate_backward()
{
    if (stdp_active) 
    {
        SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
        
        // loop over all spikes: spike = post_spike
        for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin(); spike != spikes_end; ++spike) 
        {
            // translated id of the postsynaptic neuron that spiked
            NeuronID trans_post_ind = dst->global2rank(*spike);
 
            // loop over all presynaptic partners
            for (const NeuronID* c = bkw->get_row_begin(*spike); c != bkw->get_row_end(*spike); ++c) 
            {
                #if defined(CODE_ACTIVATE_PREFETCHING_INTRINSICS) && defined(CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY)
                // prefetches next memory cells to reduce number of last-level cache misses
                _mm_prefetch((const char *)bkw->get_data_begin()[c-bkw->get_row_begin(0)+2],  _MM_HINT_NTA);
                #endif
 
                // compute plasticity update
                AurynWeight* weight = bkw->get_data(c);
                *weight += dw_post(*c, trans_post_ind, *weight, dst_cvaif->vtrace->get(trans_post_ind),
                            dst_cvaif->cexc->get(trans_post_ind), dst_cvaif->cinh->get(trans_post_ind));
 
                // clip weights if needed
                if (*weight > get_max_weight()) *weight = get_max_weight();
                if (*weight < get_min_weight()) *weight = get_min_weight();
            }
        }
    }
}

void T4wvceciMLPConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

float T4wvceciMLPConnection::forward_pre(float xpre2, float xpost1, float w, float V, float Cexc, float Cinh) {
    // ////////////////////////////DEBUG//////////////////
    // std::cout << "inside forward pre" << Wpre[0] << std::endl;
    // std::cout << ", x_pre2=" << xpre2
    //         << ", x_post1=" << xpost1
    //         << ", w=" << w
    //         << ", V=" << V
    //         << ", Cexc=" << Cexc
    //         << ", Cinh=" << Cinh << std::endl;
    // //////////////////////////////////////////////

    for (int i = 0; i < nh1; i ++){
        xh1[i] = sigmoid(rescale_trace*xpre2*Wh1[0+i*6] +
                         rescale_trace*xpost1*Wh1[1+i*6] +
                         rescale_w*w*Wh1[2+i*6] +
                         rescale_v*V*Wh1[3+i*6] +
                         rescale_cexc*Cexc*Wh1[4+i*6] +
                         rescale_cinh*Cinh*Wh1[5+i*6]);
    }

    // ////////////////////////////DEBUG//////////////////
    // std::cout << "the first layer has been computed: " << nh1 << std::endl;
    // std::cout << "nh1 " << nh1 << ", nh2 " << nh2 << ", n_coeffs_Wh1=" << n_coeffs_Wh1 << std::endl;
    // for (int i = 0; i < nh1; i ++)
    // {
    //     std::cout << xh1[i] << ", ";
    // }
    // std::cout << "; " << std::endl;
    // //////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // std::cout << " Computing second layer" << std::endl;
    // std::cout << "nh1 " << nh1 << ", nh2 " << nh2 << ", n_coeffs_Wh2=" << n_coeffs_Wh2 << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < nh2; i ++){
        xh2[i] = 0;
        for (int j = 0; j < nh1; j ++){
            ////////////////////////////////////////////////////////////////////////////
            // std::cout << "i=" << i << ", j=" << j << ", index of Wh2=" << j+i*nh1 << std::endl;
            ////////////////////////////////////////////////////////////////////////////
            xh2[i] = xh2[i] + xh1[j]*Wh2[j+i*nh1];
        }
        xh2[i] = sigmoid(xh2[i]);
    }

    // ////////////////////////////DEBUG//////////////////
    // std::cout << "the second layer has been computed: " << nh2 << std::endl;
    // for (int i = 0; i < nh2; i ++)
    // {
    //     std::cout << xh2[i] << ", ";
    // }
    // std::cout << "; " << std::endl;
    // //////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // std::cout << " Computing final layer" << std::endl;
    // std::cout << "nh1 " << nh1 << ", nh2 " << nh2 << ", n_coeffs_Wpre=" << n_coeffs_Wpre << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    float dw = 0;
    for (int j = 0; j < nh2; j ++){
        dw = dw + xh2[j]*Wpre[j];
    }
    dw = dw + Wpre[nh2]; //the bias for last layer
    ////////////////////////////////////////////////////////////////////////////
    // std::cout << " dw_pre: " << eta*dw << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    
    return(dw);
}

float T4wvceciMLPConnection::forward_post(float xpre1, float xpost2, float w, float V, float Cexc, float Cinh) {
    // ////////////////////////////DEBUG//////////////////
    // std::cout << "inside forward post" << Wpre[0] << std::endl;
    // std::cout << ", x_pre2=" << xpre1
    //         << ", x_post1=" << xpost2
    //         << ", w=" << w
    //         << ", V=" << V
    //         << ", Cexc=" << Cexc
    //         << ", Cinh=" << Cinh << std::endl;
    // //////////////////////////////////////////////

    for (int i = 0; i < nh1; i ++){
        xh1[i] = sigmoid(rescale_trace*xpre1*Wh1[0+i*6] +
                         rescale_trace*xpost2*Wh1[1+i*6] +
                         rescale_w*w*Wh1[2+i*6] +
                         rescale_v*V*Wh1[3+i*6] +
                         rescale_cexc*Cexc*Wh1[4+i*6] +
                         rescale_cinh*Cinh*Wh1[5+i*6]);
    }

    // ////////////////////////////DEBUG//////////////////
    // std::cout << "the first layer has been computed: " << nh1 << std::endl;
    // std::cout << "nh1 " << nh1 << ", nh2 " << nh2 << ", n_coeffs_Wh1=" << n_coeffs_Wh1 << std::endl;
    // for (int i = 0; i < nh1; i ++)
    // {
    //     std::cout << xh1[i] << ", ";
    // }
    // std::cout << "; " << std::endl;
    // //////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // std::cout << " Computing second layer" << std::endl;
    // std::cout << "nh1 " << nh1 << ", nh2 " << nh2 << ", n_coeffs_Wh2=" << n_coeffs_Wh2 << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < nh2; i ++){
        xh2[i] = 0;
        for (int j = 0; j < nh1; j ++){
            ////////////////////////////////////////////////////////////////////////////
            // std::cout << "i=" << i << ", j=" << j << ", index of Wh2=" << j+i*nh1 << std::endl;
            ////////////////////////////////////////////////////////////////////////////
            xh2[i] = xh2[i] + xh1[j]*Wh2[j+i*nh1];
        }
        xh2[i] = sigmoid(xh2[i]);
    }

    // ////////////////////////////DEBUG//////////////////
    // std::cout << "the second layer has been computed: " << nh2 << std::endl;
    // for (int i = 0; i < nh2; i ++)
    // {
    //     std::cout << xh2[i] << ", ";
    // }
    // std::cout << "; " << std::endl;
    // //////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // std::cout << " Computing final layer" << std::endl;
    // std::cout << "nh1 " << nh1 << ", nh2 " << nh2 << ", n_coeffs_Wpre=" << n_coeffs_Wpre << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    float dw = 0;
    for (int j = 0; j < nh2; j ++){
        dw = dw + xh2[j]*Wpost[j];
    }
    dw = dw + Wpost[nh2]; //the bias for last layer
    ////////////////////////////////////////////////////////////////////////////
    // std::cout << " dw_post: " << eta*dw << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    
    return(dw);
}

float T4wvceciMLPConnection::sigmoid(const float& z) {
    float x;
    if (z > 0.0) {
        x = 1.0 / (1.0 + std::exp(-z));
    } else {
        x = std::exp(z) / (1.0 + std::exp(z));
    }
    return x;
}