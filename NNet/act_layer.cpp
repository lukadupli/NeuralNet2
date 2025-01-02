#include "pch.h"
#include "act_layer.h"

namespace NNet {
	ActL::ActL(double lrate_, vd_F_vd ActFunc_, md_F_vd ActDeriv_) {
		id = "Act";
		lrate = lrate_;
		ActFunc = ActFunc_;
		ActDeriv = ActDeriv_;
	}
	ActL::ActL(const ActL& other) {
		id = "Act";
		lrate = other.LRate();
		bias = other.Bias();
		ActFunc = other.GetActFunc();
		ActDeriv = other.GetActDeriv();
		in_sz = other.InSize();
		out_sz = other.OutSize();
	}
	ActL::ActL(std::istream& istr) {
		id = "Act";
		Read(istr);
	}

	vd_F_vd ActL::GetActFunc() const { return ActFunc; }
	md_F_vd ActL::GetActDeriv() const { return ActDeriv; }

	Eigen::VectorXd ActL::Bias() const { return bias; }

	int ActL::InSize() const { return in_sz; }
	int ActL::OutSize() const { return out_sz; }

	void ActL::SetInputSize(int input_sz) {
		in_sz = input_sz;
		out_sz = in_sz;
	}
	void ActL::InitParams(d_F GenFunc) {
		bias = Eigen::VectorXd::Zero(in_sz);
	}

	Eigen::VectorXd ActL::Forward(const Eigen::VectorXd& in) {
		if (in.size() != in_sz) throw Exception("ActL::Forward: Input sizes don't match!");
		cache = in;
		return ActFunc(in + bias);
	}
	Eigen::VectorXd ActL::Backward(const Eigen::VectorXd& grads) {
		if (grads.size() != out_sz) throw Exception("ActL::Backward: Output sizes don't match!");

		Eigen::VectorXd ret = ActDeriv(cache + bias) * grads;
		bias -= lrate * ret;
		return ret;
	}

	std::istream& ActL::Read(std::istream& istr) {
		istr >> in_sz >> lrate >> ActFunc >> ActDeriv;
		out_sz = in_sz;

		bias = Eigen::VectorXd(in_sz);
		for (int i = 0; i < in_sz; i++) istr >> bias(i);
		return istr;
	}
	std::ostream& ActL::Write(std::ostream& ostr) const {
		ostr << id << '\n' << in_sz << ' ' << lrate << ' ' << ActFunc << ActDeriv << '\n' << bias << '\n';
		return ostr;
	}
}