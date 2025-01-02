#include "pch.h"
#include "neural_net.h"

namespace NNet {
	NeuralNet::NeuralNet(int input_sz, const std::vector<Layer*>& layers, d_F_vd_vd LossFunc, vd_F_vd_vd LossDeriv, d_F RandGen) 
	: in_sz(input_sz), layers(layers), LossFunc(LossFunc), LossDeriv(LossDeriv) 
	{
        for (auto& layer : layers) {
			layer->SetInputSize(input_sz);
			layer->InitParams(RandGen);
			input_sz = layer->OutSize();
		}

		out_sz = input_sz;
	}

	NeuralNet::NeuralNet(const NeuralNet& other) {
		in_sz = other.InSize();
		out_sz = other.OutSize();

		LossFunc = other.GetLossFunc();
		LossDeriv = other.GetLossDeriv();

		layers = other.LayersCopy();
	}

	NeuralNet::NeuralNet(std::istream& istr) {
		Load(istr);
	}

	NeuralNet::NeuralNet(const std::string& path) {
		Load(path);
	}

	NeuralNet::~NeuralNet() {
		for (auto& e : layers) delete e;
	}

	int NeuralNet::InSize() const { return in_sz; }
	int NeuralNet::OutSize() const { return out_sz; }

	d_F_vd_vd NeuralNet::GetLossFunc() const { return LossFunc; }
	vd_F_vd_vd NeuralNet::GetLossDeriv() const { return LossDeriv; }

	std::vector<Layer*> NeuralNet::LayersCopy() const {
		std::vector<Layer*> cpy;
		for (auto& layer : layers) cpy.push_back(layer->Clone());

		return cpy;
	}

	Eigen::VectorXd NeuralNet::Query(const Eigen::VectorXd& in) {
		if (in.size() != in_sz) throw Exception("NeuralNet::Query: Rececived input vector is not the right size!");

		Eigen::VectorXd ret = in;
		for (auto& layer : layers) ret = layer->Forward(ret);

		return ret;
	}
	Eigen::VectorXd NeuralNet::Query(const std::vector<double>& in) {
		return Query(Vec2Eig(in));
	}
	
	Eigen::VectorXd NeuralNet::BackQuery(const Eigen::VectorXd& grads) {
		if (grads.size() != out_sz) throw Exception("NeuralNet::BackQuery: Rececived gradients list is not the right size!");
		Eigen::VectorXd ret = grads;
		for (int i = layers.size() - 1; i >= 0; i--) ret = layers[i]->Backward(ret);

		return ret;
	}

	double NeuralNet::Fit(const Eigen::VectorXd& in, const Eigen::VectorXd& target) {
		if(in.size() != in_sz) throw Exception("NeuralNet::Fit: Rececived input vector is not the right size!");
		if(target.size() != out_sz) throw Exception("NeuralNet::Fit: Rececived target vector is not the right size!");
		Eigen::VectorXd out = Query(in);
		double loss = LossFunc(out, target);

		BackQuery(LossDeriv(out, target));

		return loss;
	}
	double NeuralNet::Fit(const std::vector<double>& in, const std::vector<double>& target) {
		return Fit(Vec2Eig(in), Vec2Eig(target));
	}

	std::istream& NeuralNet::Load(std::istream& istr) {
        for (auto& e : layers) delete e;
        layers.clear();

		int lcnt;
		istr >> lcnt >> in_sz >> out_sz >> LossFunc >> LossDeriv;

        int sz = in_sz;
		std::string id;
		for (int i = 0; i < lcnt; i++) {
			istr >> id;
            if (id == "Dense") layers.push_back(new DenseL(istr));
            else if (id == "Act") layers.push_back(new ActL(istr));
            else if (id == "Conv") layers.push_back(new ConvL(istr));
            else if (id == "Pool") layers.push_back(new PoolL(istr));
			else throw Exception("NeuralNet::Load: data in the given stream cannot be interpreted as a NeuralNet!");
		}

		return istr;
	}
	void NeuralNet::Load(const std::string& path) {
		std::ifstream istr{path};
		Load(istr);
	}

	std::ostream& NeuralNet::Save(std::ostream& ostr) const {
		ostr << layers.size() << ' ' << in_sz << ' ' << out_sz << ' ' << LossFunc << LossDeriv << '\n';
		for (auto& layer : layers) layer->Write(ostr);
		return ostr;
	}
	void NeuralNet::Save(const std::string& path) const {
		if(system(("type nul > " + path).c_str())) throw Exception("NeuralNet::Save: Invalid path name!");
		std::ofstream ostr{ path };
		Save(ostr);
	}
}