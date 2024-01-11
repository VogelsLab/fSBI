/* 
* This is a part of Synapseek, written by Basile Confavreux. 
* It uses the spiking network simulator Auryn written by Friedemann Zenke.
*/

#ifndef T4WVCECIMLPCONNECTION_H_
#define T4WVCECIMLPCONNECTION_H_

#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/DuplexConnection.h"
#include "auryn/Trace.h"
#include "auryn/LinearTrace.h"
#include "CVAIFGroup.h"
#include <vector>

namespace auryn {


/*! \brief Implements a STDP rule parameterized with an MLP.
It has 2 pre traces and 2 post traces (each with their own timescale).
Plasticity updates depend on weight, smoothed voltage, and two codependent plasticity terms Cexc and Cinh. 
Typically only the final layer is trained.
This class requires a neuron group with vtrace, cexc and cinh attributes (e.g. CVAIFGroup)
* Class adapted from SymmetricSTDPConnection
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

class T4wvceciMLPConnection : public DuplexConnection
{

public:
	int nh1;
	int nh2;

	int n_coeffs_Wh1;
	int n_coeffs_Wh2;
	int n_coeffs_Wpre;
	int n_coeffs_Wpost;

	//rescale inputs to the MLP for more useful computation with sigmoid
	float rescale_trace;
	float rescale_v;
	float rescale_w;
	float rescale_cexc;
	float rescale_cinh;

	CVAIFGroup* dst_cvaif; //pointer to destination neuron group that has to be a CVAIF group

	float eta;
	std::vector<AurynFloat> Wh1;
	std::vector<AurynFloat> Wh2;
	std::vector<AurynFloat> Wpre;
	std::vector<AurynFloat> Wpost;

	std::vector<AurynFloat> xh1;
	std::vector<AurynFloat> xh2;

	Trace * tr_pre1;
	Trace * tr_pre2;
	Trace * tr_post1;
	Trace * tr_post2;

	inline AurynWeight dw_pre(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat Cexc, AurynFloat Cinh);
	inline AurynWeight dw_post(NeuronID pre,NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat Cexc, AurynFloat Cinh);

	inline void propagate_forward();
	inline void propagate_backward();

	bool stdp_active;

	/*! Constructor to create a random sparse connection object and set up plasticity.
	 * @param sourceAurynWeight  the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param weight the initial weight of all connections.
	 * @param sparseness the connection probability for the sparse random set-up of the connections.
	 * @param coeffs parameters of the learning rule: [tau_pre1, tau_pre2, tau_post1, tau_post2, coeffs_pre[i].., coeffs_post[i]...]
	 * @param nh1 number of neurons of first hidden layer
	 * @param nh2 number of neurons of second hidden layer
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	T4wvceciMLPConnection(CVAIFGroup* source, CVAIFGroup* destination, 
			AurynWeight weight, AurynFloat sparseness, std::vector<float> coeffs, int nh1_, int nh2_,
			AurynWeight maxweight=1.5 , TransmitterType transmitter=GABA, string name="T4wvceciMLPConnection");

	virtual ~T4wvceciMLPConnection();
	void init(std::vector<float> coeffs, AurynWeight maxweight, int nh1_, int nh2_);
	void free();

	virtual void propagate();

	float forward_pre(float xpre2,
					float xpost1,
                    float w, 
                    float V, 
                    float Cexc,
					float Cinh);

	float forward_post(float xpre1,
					float xpost2,
                    float w, 
                    float V, 
                    float Cexc,
					float Cinh);

	float sigmoid(const float& x);

};

}

#endif /*T4WVCECIMLPCONNECTION_H_*/